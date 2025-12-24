#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import enum
import warnings

import cupy as cp
import numpy as np

_PACK_SLICE_KERNELS = {}


def _get_pack_slice_kernels(dtype):
    """Lazy-create CuPy RawKernels used by the slice fast-path packing."""
    key = str(np.dtype(dtype))
    k = _PACK_SLICE_KERNELS.get(key)
    if k is not None:
        return k

    if np.dtype(dtype) == np.float32:
        ctype = "float"
    elif np.dtype(dtype) == np.float64:
        ctype = "double"
    else:
        raise TypeError(f"Unsupported dtype for slice packing: {dtype}")

    code = f"""
    extern "C" __global__
    void pack_X(const {ctype}* X,
                long long x_s0, long long x_s1,
                int n_cols,
                const long long* starts,
                const long long* lens,
                {ctype}* out,
                long long o_s0, long long o_s1, long long o_s2)
    {{
      int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
      int c = (int)(blockIdx.y * blockDim.y + threadIdx.y);
      int m = (int)blockIdx.z;
      long long len = lens[m];
      if (r >= len || c >= n_cols) return;
      long long start = starts[m];
      out[(long long)r * o_s0 + (long long)c * o_s1 + (long long)m * o_s2] =
        X[(start + (long long)r) * x_s0 + (long long)c * x_s1];
    }}

    extern "C" __global__
    void pack_y(const {ctype}* y,
                long long y_s0,
                const long long* starts,
                const long long* lens,
                {ctype}* out,
                long long o_s0, long long o_s1)
    {{
      int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
      int m = (int)blockIdx.z;
      long long len = lens[m];
      if (r >= len) return;
      long long start = starts[m];
      out[(long long)r * o_s0 + (long long)m * o_s1] =
        y[(start + (long long)r) * y_s0];
    }}
    """
    kx = cp.RawKernel(code, "pack_X")
    ky = cp.RawKernel(code, "pack_y")
    _PACK_SLICE_KERNELS[key] = (kx, ky)
    return kx, ky

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray, cuda_ptr
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.internals.outputs import reflect
from cuml.linear_model.base import LinearPredictMixin

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t
from libc.stdint cimport int32_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef void olsFit(handle_t& handle,
                     float *input,
                     size_t n_rows,
                     size_t n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     int algo,
                     float *sample_weight) except +

    cdef void olsFit(handle_t& handle,
                     double *input,
                     size_t n_rows,
                     size_t n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     int algo,
                     double *sample_weight) except +

    cdef void olsFitDeviceIntercept(handle_t& handle,
                                   float *input,
                                   size_t n_rows,
                                   size_t n_cols,
                                   float *labels,
                                   float *coef,
                                   float *intercept_device,
                                   bool fit_intercept,
                                   int algo,
                                   float *sample_weight) except +

    cdef void olsFitDeviceIntercept(handle_t& handle,
                                   double *input,
                                   size_t n_rows,
                                   size_t n_cols,
                                   double *labels,
                                   double *coef,
                                   double *intercept_device,
                                   bool fit_intercept,
                                   int algo,
                                   double *sample_weight) except +

    cdef void olsFitDeviceInterceptWorkspace(handle_t& handle,
                                            float *input,
                                            size_t n_rows,
                                            size_t n_cols,
                                            float *labels,
                                            float *coef,
                                            float *intercept_device,
                                            float *mu_input,
                                            float *mu_labels,
                                            bool fit_intercept,
                                            int algo,
                                            float *sample_weight) except +

    cdef void olsFitDeviceInterceptWorkspace(handle_t& handle,
                                            double *input,
                                            size_t n_rows,
                                            size_t n_cols,
                                            double *labels,
                                            double *coef,
                                            double *intercept_device,
                                            double *mu_input,
                                            double *mu_labels,
                                            bool fit_intercept,
                                            int algo,
                                            double *sample_weight) except +

    cdef void olsFitDeviceInterceptWorkspaceAsyncInfo(handle_t& handle,
                                                     float *input,
                                                     size_t n_rows,
                                                     size_t n_cols,
                                                     float *labels,
                                                     float *coef,
                                                     float *intercept_device,
                                                     float *mu_input,
                                                     float *mu_labels,
                                                     int *dev_info_out,
                                                     bool fit_intercept,
                                                     int algo,
                                                     float *sample_weight) except +

    cdef void olsFitDeviceInterceptWorkspaceAsyncInfo(handle_t& handle,
                                                     double *input,
                                                     size_t n_rows,
                                                     size_t n_cols,
                                                     double *labels,
                                                     double *coef,
                                                     double *intercept_device,
                                                     double *mu_input,
                                                     double *mu_labels,
                                                     int *dev_info_out,
                                   bool fit_intercept,
                                   int algo,
                                   double *sample_weight) except +


class Algo(enum.IntEnum):
    """The lstsq solver algorithm"""
    SVD = 0
    EIG = 1
    QR = 2
    SVD_QR = 3

    @classmethod
    def parse(cls, name):
        out = {
            "svd": cls.SVD,
            "eig": cls.EIG,
            "qr": cls.QR,
            "svd-qr": cls.SVD_QR,
            "svd-jacobi": cls.SVD
        }.get(name)
        if out is None:
            raise ValueError("algorithm {name!r} is not supported")
        return out


# 1e-10 chosen to match C++ implementation
_divide_non_zero = cp.ElementwiseKernel(
    "T x, T y",
    "T z",
    "z = abs(y) < 1e-10 ? x : x / y",
    "divide_non_zero"
)


class LinearRegression(Base,
                       InteropMixin,
                       LinearPredictMixin,
                       RegressorMixin,
                       FMajorInputTagMixin):
    """
    LinearRegression is a simple machine learning model where the response y is
    modelled by a linear combination of the predictors in X.

    cuML's LinearRegression can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or `__cuda_array_interface__`
    compliant), in addition to cuDF objects.
    It provides two algorithms: Singular Value
    Decomposition (SVD) and Eigndecomposition (Eig) to fit a linear model.
    SVD is more numerically stable, but Eig (the default) is much faster.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> import cudf

        >>> # Both import methods supported
        >>> from cuml import LinearRegression
        >>> from cuml.linear_model import LinearRegression
        >>> lr = LinearRegression(fit_intercept = True, algorithm = "eig")
        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([1,1,2,2], dtype=cp.float32)
        >>> X['col2'] = cp.array([1,2,2,3], dtype=cp.float32)
        >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))
        >>> reg = lr.fit(X,y)
        >>> print(reg.coef_)
        0   1.0
        1   2.0
        dtype: float32
        >>> print(reg.intercept_)
        3.0...

        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
        >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)
        >>> preds = lr.predict(X_new)
        >>> print(preds) # doctest: +SKIP
        0   15.999...
        1   14.999...
        dtype: float32


    Parameters
    ----------
    algorithm : {'auto', 'svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi'}, (default = 'auto')
        Choose an algorithm:

          * 'auto' - ``'eig'``, or ``'svd'`` if y multi-target or X has only one column
          * ``'svd'`` - alias for svd-jacobi
          * ``'eig'`` - use an eigendecomposition of the covariance matrix
          * ``'qr'``  - use QR decomposition algorithm and solve `Rx = Q^T y`
          * ``'svd-qr'`` - compute SVD decomposition using QR algorithm
          * ``'svd-jacobi'`` - compute SVD decomposition using Jacobi iterations

        Among these algorithms, only ``'svd-jacobi'`` supports the case when the
        number of features is larger than the sample size; this algorithm
        is force-selected automatically in such a case.

        For the broad range of inputs, ``'eig'`` and ``'qr'`` are usually the fastest,
        followed by ``'svd-jacobi'`` and then ``'svd-qr'``. In theory, `svd`-based
        algorithms are more numerically stable.
    fit_intercept : boolean (default = True)
        If True, LinearRegression tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    copy_X : boolean, default=True
        If True, it is guaranteed that a copy of X is created, leaving the
        original X unchanged. However, if set to False, X may be modified
        directly, which would reduce the memory usage of the estimator.

        .. versionchanged:: 23.08
            Starting from version 23.08, the new `copy_X` parameter defaults
            to ``True``, ensuring a copy of X is created after passing it to
            `fit()`, preventing any changes to the input, but with increased
            memory usage. This represents a change in behavior from previous
            versions. With `copy_X=False` a copy might still be created if
            necessary.
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    LinearRegression suffers from multicollinearity (when columns are
    correlated with each other), and variance explosions from outliers.
    Consider using :class:`Ridge` to fix the multicollinearity problem, and
    consider maybe first :class:`DBSCAN` to remove the outliers, or
    statistical analysis to filter possible outliers.

    **Applications of LinearRegression**

        LinearRegression is used in regression tasks where one wants to predict
        say sales or house prices. It is also used in extrapolation or time
        series tasks, dynamic systems modelling and many other machine learning
        tasks. This model should be first tried if the machine learning problem
        is a regression task (predicting a continuous variable).

    For additional information, see scikit-learn's documentation for
    :class:`sklearn.linear_model.LinearRegression`.

    For an additional example see `the OLS notebook
    <https://github.com/rapidsai/cuml/blob/main/notebooks/linear_regression_demo.ipynb>`__.
    """

    coef_ = CumlArrayDescriptor(order="F")
    intercept_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.linear_model.LinearRegression"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "algorithm",
            "fit_intercept",
            "copy_X",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        return {
            "fit_intercept": model.fit_intercept,
            "copy_X": model.copy_X,
        }

    def _params_to_cpu(self):
        return {
            "fit_intercept": self.fit_intercept,
            "copy_X": self.copy_X,
        }

    def _attrs_from_cpu(self, model):
        return {
            "intercept_": to_gpu(model.intercept_, order="F"),
            "coef_": to_gpu(model.coef_, order="F"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "intercept_": to_cpu(self.intercept_),
            "coef_": to_cpu(self.coef_),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        algorithm="auto",
        fit_intercept=True,
        copy_X=True,
        handle=None,
        verbose=False,
        output_type=None
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        self.algorithm = algorithm
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

    def _select_algo(self, X, y):
        """Select the solver algorithm based on `algorithm` and problem dimensions"""
        if X.shape[0] == 1:
            fallback_reason = "single-column X"
        elif y.ndim == 2 and y.shape[1] > 1:
            fallback_reason = "multi-column y"
        else:
            fallback_reason = None

        if self.algorithm == "auto":
            algo = Algo.SVD if fallback_reason else Algo.EIG
        else:
            algo = Algo.parse(self.algorithm)
            if fallback_reason and algo != Algo.SVD:
                warnings.warn(
                    (
                        "Falling back to `algorithm='svd'` as `algorithm="
                        "{self.algorithm!r}` doesn't support {fallback_reason}."
                    ),
                    UserWarning,
                )
                algo = Algo.SVD
        return algo

    @generate_docstring()
    @reflect(reset=True)
    def fit(self, X, y, sample_weight=None, *, convert_dtype=True) -> "LinearRegression":
        """
        Fit the model with X and y.

        """
        X_m = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        ).array

        if X_m.shape[0] < 2:
            raise ValueError("X matrix must have at least two rows")

        if X_m.shape[1] < 1:
            raise ValueError("X matrix must have at least one column")

        y_m = input_to_cuml_array(
            y,
            check_dtype=X_m.dtype,
            convert_to_dtype=(X_m.dtype if convert_dtype else None),
            check_rows=X_m.shape[0],
            order="F",
        ).array

        if sample_weight is not None:
            # Always copy the weights, all solvers mutate them
            sample_weight = input_to_cuml_array(
                sample_weight,
                check_dtype=X_m.dtype,
                convert_to_dtype=(X_m.dtype if convert_dtype else None),
                check_rows=X_m.shape[0],
                check_cols=1,
                order="F",
                deepcopy=True,
            ).array

        cdef int algo = self._select_algo(X_m, y_m)

        X_is_copy = cuda_ptr(X) != X_m.ptr
        y_is_copy = cuda_ptr(y) != y_m.ptr

        if y_m.ndim > 1 and y_m.shape[1] > 1:
            # Fallback to cupy SVD implementation for multi-target problems
            self._fit_multi_target(
                X_m, y_m, sample_weight, X_is_copy=X_is_copy, y_is_copy=y_is_copy
            )
            return self

        # All libcuml solvers mutate the inputs. Here we make a copy requested
        # (and one wasn't already made).
        if not X_is_copy and self.copy_X:
            X_m = input_to_cuml_array(X_m, deepcopy=True).array
        if not y_is_copy:
            y_m = input_to_cuml_array(y_m, deepcopy=True).array

        coef = CumlArray.zeros(X_m.shape[1], dtype=X_m.dtype)

        cdef size_t n_rows = X_m.shape[0]
        cdef size_t n_cols = X_m.shape[1]
        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t y_ptr = y_m.ptr
        cdef uintptr_t sample_weight_ptr = (
            0 if sample_weight is None else sample_weight.ptr
        )
        cdef uintptr_t coef_ptr = coef.ptr
        cdef bool is_float32 = X_m.dtype == np.float32
        cdef float intercept_f32
        cdef double intercept_f64
        # Always use 2 streams to expose concurrency in the eig computation
        handle = get_handle(model=self, n_streams=2)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef bool fit_intercept = self.fit_intercept

        with nogil:
            if is_float32:
                olsFit(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>y_ptr,
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    algo,
                    <float*>sample_weight_ptr,
                )
            else:
                olsFit(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>y_ptr,
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    algo,
                    <double*>sample_weight_ptr,
                )
        handle.sync()

        self.intercept_ = intercept_f32 if is_float32 else intercept_f64
        self.coef_ = coef

        return self

    def fit_many(
        self,
        train_pairs,
        *,
        max_concurrency=8,
        eval_pairs=None,
        scoring="r2",
        return_models=True,
        return_scores=False,
        convert_dtype=True,
    ):
        """Fit many independent OLS models concurrently on a single GPU.

        Parameters
        ----------
        train_pairs : iterable
            Iterable of (X_train, y_train) pairs.
        max_concurrency : int, default=8
            Max number of CUDA streams/RAFT handles to use concurrently.
        eval_pairs : iterable, optional
            Iterable of (X_eval, y_eval) pairs. If provided and return_scores=True,
            a score is computed per model after fitting.
        scoring : str or callable, default='r2'
            Scoring to use if `return_scores=True` and `eval_pairs` is provided.
        return_models : bool, default=True
            If True, return list of fitted estimator objects.
        return_scores : bool, default=False
            If True, return scores (requires `eval_pairs`).
        convert_dtype : bool, default=True
            Whether to convert input data to float32/float64 as in `.fit()`.

        Returns
        -------
        models : list[LinearRegression]
            Returned if return_models=True.
        scores : array-like
            Returned if return_scores=True.
        """
        train_pairs = list(train_pairs)
        n_models = len(train_pairs)
        if n_models == 0:
            if return_models and return_scores:
                return [], cp.asarray([], dtype=cp.float32)
            elif return_models:
                return []
            else:
                return cp.asarray([], dtype=cp.float32)

        if max_concurrency is None:
            max_concurrency = n_models
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        n_workers = min(max_concurrency, n_models)

        if eval_pairs is not None:
            eval_pairs = list(eval_pairs)
            if len(eval_pairs) != n_models:
                raise ValueError("eval_pairs must have the same length as train_pairs")
        if return_scores and eval_pairs is None:
            raise ValueError("return_scores=True requires eval_pairs")
        if scoring != "r2":
            raise NotImplementedError("Only scoring='r2' is supported in fit_many for now")

        # Create independent handles. Use a small stream pool so RAFT internals (e.g., lstsqEig)
        # can overlap independent gemm/gemv work without forcing global synchronizations.
        handles = [Handle(n_streams=2) for _ in range(n_workers)]

        # Cython typed locals must be declared before use (including in slice fast-path packing).
        cdef handle_t* handle_
        cdef uintptr_t stream_ptr
        cdef int algo
        cdef size_t n_rows
        cdef size_t n_cols
        cdef uintptr_t X_ptr
        cdef uintptr_t y_ptr
        cdef uintptr_t coef_ptr
        cdef uintptr_t intercept_ptr
        cdef uintptr_t mu_input_ptr
        cdef uintptr_t mu_labels_ptr
        cdef uintptr_t info_ptr
        cdef uintptr_t sample_weight_ptr
        cdef bool is_float32
        cdef bool fit_intercept

        # Slice fast-path: train_pairs items may be (X_base, y_base, slice) to avoid per-window materialization.
        slice_mode = (
            n_models > 0
            and isinstance(train_pairs[0], tuple)
            and len(train_pairs[0]) == 3
            and isinstance(train_pairs[0][2], slice)
        )
        if slice_mode:
            X_base0, y_base0, _sl0 = train_pairs[0]
            for t in train_pairs:
                if not (isinstance(t, tuple) and len(t) == 3 and isinstance(t[2], slice)):
                    slice_mode = False
                    break
                if t[0] is not X_base0 or t[1] is not y_base0:
                    # Keep v1 simple: require a single shared base X/y.
                    slice_mode = False
                    break

        # Pre-scan shapes so we can allocate reusable per-worker buffers (avoids cudaMalloc/cudaFree churn).
        # This assumes all problems have the same number of features, which is typical for walk-forward CV.
        cdef size_t max_rows = 0
        cdef size_t common_cols = 0
        work_dtype = None
        for item in train_pairs:
            if slice_mode:
                X_i, y_i, sl = item
            else:
                X_i, y_i = item
            # Avoid `input_to_cuml_array` during the prescan: it can trigger tiny device copies
            # (dtype/order conversion) and we only need shapes here.
            if common_cols == 0:
                common_cols = <size_t>X_i.shape[1]
            elif <size_t>X_i.shape[1] != common_cols:
                raise ValueError(
                    "fit_many requires a constant number of features across all train_pairs"
                )

            if slice_mode:
                if sl.step not in (None, 1):
                    raise ValueError("slice fast-path requires step=None or 1")
                start = 0 if sl.start is None else int(sl.start)
                stop = int(sl.stop) if sl.stop is not None else int(X_i.shape[0])
                n_r = max(0, stop - start)
                if <size_t>n_r > max_rows:
                    max_rows = <size_t>n_r
            else:
                if <size_t>X_i.shape[0] > max_rows:
                    max_rows = <size_t>X_i.shape[0]

            if work_dtype is None:
                if convert_dtype:
                    # Match `.fit()` behavior when convert_dtype=True:
                    # - Preserve float32/float64 inputs
                    # - Convert other numeric types to float32
                    dt = getattr(X_i, "dtype", None)
                    if dt is None and hasattr(X_i, "dtypes"):
                        # cudf.DataFrame path
                        dts = X_i.dtypes
                        dt = dts.iloc[0] if hasattr(dts, "iloc") else dts[0]
                    if dt in (np.float32, np.float64):
                        work_dtype = dt
                    else:
                        work_dtype = np.float32
                else:
                    # Keep input dtype (must be float32/float64 to preserve libcuml behavior).
                    dt = getattr(X_i, "dtype", None)
                    if dt is None and hasattr(X_i, "dtypes"):
                        # cudf.DataFrame path
                        dts = X_i.dtypes
                        dt = dts.iloc[0] if hasattr(dts, "iloc") else dts[0]
                    if dt not in (np.float32, np.float64):
                        raise ValueError(
                            "fit_many requires float32/float64 inputs when convert_dtype=False"
                        )
                    work_dtype = dt

        if common_cols == 0 or max_rows < 2:
            raise ValueError("Invalid train_pairs inputs")

        # Allocate per-worker scratch buffers once (and cache them across calls).
        cache = getattr(self, "_fit_many_worker_cache", None)
        reuse_ok = (
            cache is not None
            and cache.get("dtype") == work_dtype
            and cache.get("common_cols") == int(common_cols)
            and cache.get("n_workers") == int(n_workers)
            and cache.get("max_rows", 0) >= int(max_rows)
        )

        if reuse_ok:
            work_X = cache["work_X"]
            work_y = cache["work_y"]
            work_mu_input = cache["work_mu_input"]
            work_mu_labels = cache["work_mu_labels"]
        else:
            # We copy each window into scratch on that worker's stream so the solver can safely mutate
            # without touching user data.
            work_X = []
            work_y = []
            work_mu_input = []
            work_mu_labels = []
            for w in range(n_workers):
                work_X.append(CumlArray.empty((max_rows, common_cols), dtype=work_dtype, order="F"))
                work_y.append(CumlArray.empty((max_rows,), dtype=work_dtype, order="F"))
                # Workspace for fit_intercept centering (avoids per-fit device allocations)
                work_mu_input.append(CumlArray.empty((common_cols,), dtype=work_dtype, order="F"))
                work_mu_labels.append(CumlArray.empty((1,), dtype=work_dtype, order="F"))
            self._fit_many_worker_cache = {
                "dtype": work_dtype,
                "max_rows": int(max_rows),
                "common_cols": int(common_cols),
                "n_workers": int(n_workers),
                "work_X": work_X,
                "work_y": work_y,
                "work_mu_input": work_mu_input,
                "work_mu_labels": work_mu_labels,
            }

        # Allocate outputs in one contiguous block to avoid per-model allocations.
        # IMPORTANT: store as (n_features, n_models) in F-order so each model's coef_ is contiguous.
        out_coef = CumlArray.empty((common_cols, n_models), dtype=work_dtype, order="F")
        out_intercepts = CumlArray.empty((n_models,), dtype=work_dtype, order="F")
        # Per-fit solver info (2 ints per model: primary + secondary for QR)
        out_info = CumlArray.empty((n_models, 2), dtype=np.int32, order="C")
        out_info_cupy = out_info.to_output("cupy")

        # Spawn per-fit estimators with the same hyperparameters.
        models = []
        if return_models:
            for i in range(n_models):
                models.append(
                    LinearRegression(
                        algorithm=self.algorithm,
                        fit_intercept=self.fit_intercept,
                        copy_X=self.copy_X,
                        handle=handles[i % n_workers],
                        verbose=self.verbose,
                        output_type=self.output_type,
                    )
                )

        # Store outputs for later materialization
        coefs = [None] * n_models
        intercept_devs = [None] * n_models
        dtypes = [None] * n_models
        n_features = [None] * n_models
        # Keep temporary converted (safe-to-mutate) inputs alive until all streams finish.
        tmp_keepalive = []

        # Slice fast-path packing state
        pack_X_cupy = None
        pack_y_cupy = None
        local_pos_for_model = None
        base_keepalive = None

        if slice_mode:
            # Convert base arrays to CuPy once. For non-CuPy inputs this may allocate,
            # but avoids per-window conversion/copy overhead.
            if hasattr(X_base0, "to_cupy"):
                X_base_cp = X_base0.to_cupy()
            elif hasattr(X_base0, "to_output"):
                X_base_cp = X_base0.to_output("cupy")
            else:
                X_base_cp = cp.asarray(X_base0)

            if hasattr(y_base0, "to_cupy"):
                y_base_cp = y_base0.to_cupy()
            elif hasattr(y_base0, "to_output"):
                y_base_cp = y_base0.to_output("cupy")
            else:
                y_base_cp = cp.asarray(y_base0)
            y_base_cp = y_base_cp.reshape(-1)

            # Ensure dtype matches work_dtype (preserve float64 when requested).
            if X_base_cp.dtype != work_dtype:
                X_base_cp = X_base_cp.astype(work_dtype, copy=True, order="A")
            if y_base_cp.dtype != work_dtype:
                y_base_cp = y_base_cp.astype(work_dtype, copy=True, order="A")
            base_keepalive = (X_base_cp, y_base_cp)

            # Build per-worker model lists and slice parameters.
            worker_ids = [[] for _ in range(n_workers)]
            starts_by_worker = [[] for _ in range(n_workers)]
            lens_by_worker = [[] for _ in range(n_workers)]
            n_rows_by_model = [0] * n_models
            local_pos_for_model = [0] * n_models

            for i, (_Xb, _yb, sl) in enumerate(train_pairs):
                worker = i % n_workers
                start = 0 if sl.start is None else int(sl.start)
                stop = int(sl.stop) if sl.stop is not None else int(X_base_cp.shape[0])
                if sl.step not in (None, 1):
                    raise ValueError("slice fast-path requires step=None or 1")
                n_r = max(0, stop - start)
                n_rows_by_model[i] = n_r
                local_pos_for_model[i] = len(worker_ids[worker])
                worker_ids[worker].append(i)
                starts_by_worker[worker].append(start)
                lens_by_worker[worker].append(n_r)

            # Allocate packed buffers per worker (3D for X, 2D for y).
            pack_X_cupy = [None] * n_workers
            pack_y_cupy = [None] * n_workers
            for w in range(n_workers):
                k = len(worker_ids[w])
                if k == 0:
                    continue
                Xp = CumlArray.empty((max_rows, common_cols, k), dtype=work_dtype, order="F").to_output("cupy")
                yp = CumlArray.empty((max_rows, k), dtype=work_dtype, order="F").to_output("cupy")
                pack_X_cupy[w] = Xp
                pack_y_cupy[w] = yp

                starts_d = cp.asarray(starts_by_worker[w], dtype=cp.int64)
                lens_d = cp.asarray(lens_by_worker[w], dtype=cp.int64)

                kx, ky = _get_pack_slice_kernels(work_dtype)
                # Strides in elements (not bytes)
                x_s0 = X_base_cp.strides[0] // X_base_cp.itemsize
                x_s1 = X_base_cp.strides[1] // X_base_cp.itemsize
                o_s0 = Xp.strides[0] // Xp.itemsize
                o_s1 = Xp.strides[1] // Xp.itemsize
                o_s2 = Xp.strides[2] // Xp.itemsize
                y_s0 = y_base_cp.strides[0] // y_base_cp.itemsize
                yo_s0 = yp.strides[0] // yp.itemsize
                yo_s1 = yp.strides[1] // yp.itemsize

                # Launch packing on the worker stream so it is ordered before the fit calls.
                handle_py = handles[w]
                handle_ = <handle_t*><size_t>handle_py.getHandle()
                stream_ptr = <uintptr_t>handle_[0].get_stream().value()
                with cp.cuda.ExternalStream(int(stream_ptr)):
                    # pack X: grid (rows, cols, k)
                    tx, ty = 32, 4
                    grid_x = (int(max_rows) + tx - 1) // tx
                    grid_y = (int(common_cols) + ty - 1) // ty
                    kx((grid_x, grid_y, k),
                       (tx, ty, 1),
                       (X_base_cp, x_s0, x_s1, int(common_cols), starts_d, lens_d, Xp, o_s0, o_s1, o_s2))
                    # pack y: grid (rows, 1, k)
                    tyb = 256
                    grid_yb = (int(max_rows) + tyb - 1) // tyb
                    ky((grid_yb, 1, k),
                       (tyb, 1, 1),
                       (y_base_cp, y_s0, starts_d, lens_d, yp, yo_s0, yo_s1))

        # Enqueue fits
        for i, item in enumerate(train_pairs):
            worker = i % n_workers
            handle_py = handles[worker]
            handle_ = <handle_t*><size_t>handle_py.getHandle()
            # pylibraft handle_t.get_stream() returns an rmm::cuda_stream_view; cupy needs cudaStream_t.
            stream_ptr = <uintptr_t>handle_[0].get_stream().value()

            # Ensure conversions/copies happen on the same CUDA stream as the fit.
            with cp.cuda.ExternalStream(int(stream_ptr)):
                if slice_mode:
                    # Inputs are already packed into per-worker buffers.
                    _Xb, _yb, sl = item
                    # Compute n_rows for this model
                    start = 0 if sl.start is None else int(sl.start)
                    stop = int(sl.stop) if sl.stop is not None else int(base_keepalive[0].shape[0])
                    n_r = max(0, stop - start)
                    if n_r < 2:
                        raise ValueError("X matrix must have at least two rows")
                    n_rows = <size_t>n_r
                    n_cols = common_cols

                    pos = local_pos_for_model[i]
                    Xw_view = pack_X_cupy[worker][:n_rows, :n_cols, pos]
                    yw_view = pack_y_cupy[worker][:n_rows, pos]
                    X_ptr = Xw_view.data.ptr
                    y_ptr = yw_view.data.ptr

                    # algorithm selection based on shapes only
                    algo = self._select_algo(Xw_view, yw_view)
                    coef = out_coef[:, i]
                    intercept_dev = out_intercepts[i : i + 1]
                    info_ptr = <uintptr_t>(out_info_cupy.data.ptr + i * 2 * sizeof(int32_t))
                    coefs[i] = coef
                    intercept_devs[i] = intercept_dev
                    dtypes[i] = work_dtype
                    n_features[i] = n_cols
                    coef_ptr = coef.ptr
                    intercept_ptr = intercept_dev.ptr
                    mu_input_ptr = work_mu_input[worker].ptr
                    mu_labels_ptr = work_mu_labels[worker].ptr
                    sample_weight_ptr = 0
                    is_float32 = (work_dtype == np.float32)
                    fit_intercept = self.fit_intercept

                    with nogil:
                        if is_float32:
                            olsFitDeviceInterceptWorkspaceAsyncInfo(
                                handle_[0],
                                <float*>X_ptr,
                                n_rows,
                                n_cols,
                                <float*>y_ptr,
                                <float*>coef_ptr,
                                <float*>intercept_ptr,
                                <float*>mu_input_ptr,
                                <float*>mu_labels_ptr,
                                <int*>info_ptr,
                                fit_intercept,
                                algo,
                                <float*>sample_weight_ptr,
                            )
                        else:
                            olsFitDeviceInterceptWorkspaceAsyncInfo(
                                handle_[0],
                                <double*>X_ptr,
                                n_rows,
                                n_cols,
                                <double*>y_ptr,
                                <double*>coef_ptr,
                                <double*>intercept_ptr,
                                <double*>mu_input_ptr,
                                <double*>mu_labels_ptr,
                                <int*>info_ptr,
                                fit_intercept,
                                algo,
                                <double*>sample_weight_ptr,
                            )
                    continue

                X_i, y_i = item
                # Avoid `input_to_cuml_array`: it may introduce extra small copies for dtype/order
                # conversion before we pack into scratch anyway.
                if hasattr(X_i, "to_cupy"):
                    X_cp = X_i.to_cupy()
                elif hasattr(X_i, "to_output"):
                    X_cp = X_i.to_output("cupy")
                else:
                    X_cp = cp.asarray(X_i)

                if X_cp.shape[0] < 2:
                    raise ValueError("X matrix must have at least two rows")
                if X_cp.shape[1] < 1:
                    raise ValueError("X matrix must have at least one column")
                if <size_t>X_cp.shape[1] != common_cols:
                    raise ValueError(
                        "fit_many requires a constant number of features across all train_pairs"
                    )

                if hasattr(y_i, "to_cupy"):
                    y_cp = y_i.to_cupy()
                elif hasattr(y_i, "to_output"):
                    y_cp = y_i.to_output("cupy")
                else:
                    y_cp = cp.asarray(y_i)
                if y_cp.shape[0] != X_cp.shape[0]:
                    raise ValueError("y must have the same number of rows as X")

                # NOTE: `sample_weight` per-model is not yet supported in v1.
                sample_weight = None

                # Validate dtype constraints. If convert_dtype=True we cast into scratch.
                if not convert_dtype and X_cp.dtype != work_dtype:
                    raise ValueError("fit_many requires a constant dtype across all train_pairs")

                algo = self._select_algo(X_cp, y_cp)
                if y_cp.ndim > 1 and y_cp.shape[1] > 1:
                    raise ValueError(
                        "fit_many currently supports only single-target y; "
                        "use algorithm='svd' and sequential .fit() for multi-target."
                    )

                # Views into the preallocated output buffers (no per-model allocations)
                coef = out_coef[:, i]
                intercept_dev = out_intercepts[i : i + 1]
                info_ptr = <uintptr_t>(out_info_cupy.data.ptr + i * 2 * sizeof(int32_t))

                coefs[i] = coef
                intercept_devs[i] = intercept_dev
                dtypes[i] = work_dtype
                n_features[i] = X_cp.shape[1]

                n_rows = X_cp.shape[0]
                n_cols = X_cp.shape[1]
                # If dtype conversion is required, use the same conversion path as `.fit()` to preserve
                # bit-for-bit behavior (notably for float64 inputs). The returned arrays are safe to mutate.
                if convert_dtype and (X_cp.dtype != work_dtype or y_cp.dtype != work_dtype):
                    X_m = input_to_cuml_array(
                        X_i,
                        convert_to_dtype=work_dtype,
                        check_dtype=[np.float32, np.float64],
                        order="F",
                        deepcopy=True,
                    ).array
                    y_m = input_to_cuml_array(
                        y_i,
                        check_dtype=work_dtype,
                        convert_to_dtype=work_dtype,
                        check_rows=X_m.shape[0],
                        order="F",
                        deepcopy=True,
                    ).array
                    tmp_keepalive.append((X_m, y_m))
                    X_ptr = X_m.ptr
                    y_ptr = y_m.ptr
                else:
                    # Copy inputs into per-worker scratch so solver can safely mutate, without touching user data.
                    Xw = work_X[worker].to_output("cupy")
                    yw = work_y[worker].to_output("cupy")
                    Xw_view = Xw[:n_rows, :n_cols]
                    yw_view = yw[:n_rows]
                    # For non-contig window slices, this is an unavoidable gather copy; keep it single-pass.
                    cp.copyto(Xw_view, X_cp)
                    cp.copyto(yw_view, y_cp.reshape(-1))
                    X_ptr = Xw_view.data.ptr
                    y_ptr = yw_view.data.ptr
                coef_ptr = coef.ptr
                intercept_ptr = intercept_dev.ptr
                mu_input_ptr = work_mu_input[worker].ptr
                mu_labels_ptr = work_mu_labels[worker].ptr
                sample_weight_ptr = 0
                is_float32 = (work_dtype == np.float32)
                fit_intercept = self.fit_intercept

                with nogil:
                    if is_float32:
                        olsFitDeviceInterceptWorkspaceAsyncInfo(
                            handle_[0],
                            <float*>X_ptr,
                            n_rows,
                            n_cols,
                            <float*>y_ptr,
                            <float*>coef_ptr,
                            <float*>intercept_ptr,
                            <float*>mu_input_ptr,
                            <float*>mu_labels_ptr,
                            <int*>info_ptr,
                            fit_intercept,
                            algo,
                            <float*>sample_weight_ptr,
                        )
                    else:
                        olsFitDeviceInterceptWorkspaceAsyncInfo(
                            handle_[0],
                            <double*>X_ptr,
                            n_rows,
                            n_cols,
                            <double*>y_ptr,
                            <double*>coef_ptr,
                            <double*>intercept_ptr,
                            <double*>mu_input_ptr,
                            <double*>mu_labels_ptr,
                            <int*>info_ptr,
                            fit_intercept,
                            algo,
                            <double*>sample_weight_ptr,
                        )

        # Synchronize once per worker handle.
        for h in handles:
            h.sync()

        # Validate solver status once (avoids per-fit host syncs).
        info_host = out_info.to_output("numpy")
        # For QR (algo=2): both entries must be 0. For EIG (algo=1): first entry must be 0.
        # For SVD paths, reference implementation does not report devInfo; they remain 0.
        if (info_host != 0).any():
            # Report first failing model and the two info values.
            import numpy as _np
            bad = _np.argwhere(info_host != 0)[0]
            i_bad = int(bad[0])
            raise RuntimeError(
                f"fit_many solver failure at model {i_bad}: dev_info={info_host[i_bad].tolist()}"
            )

        # Materialize fitted estimators and optional scores.
        out_scores = None
        if return_scores:
            out_scores = cp.empty(n_models, dtype=cp.float64)

        for i in range(n_models):
            intercept_val = intercept_devs[i].to_output("cupy")[0].item()
            if return_models:
                m = models[i]
                # Mimic @reflect(reset=True) behavior for each fit
                m._set_output_type(train_pairs[i][0])
                m._set_n_features_in(n_features[i])
                m.coef_ = coefs[i]
                m.intercept_ = intercept_val

            if return_scores:
                X_eval, y_eval = eval_pairs[i]
                if return_models:
                    # Use estimator's existing scoring API (R2)
                    out_scores[i] = models[i].score(X_eval, y_eval)
                else:
                    # If not returning models, compute score via a temporary model
                    tmp = LinearRegression(
                        algorithm=self.algorithm,
                        fit_intercept=self.fit_intercept,
                        copy_X=self.copy_X,
                        handle=handles[i % n_workers],
                        verbose=self.verbose,
                        output_type=self.output_type,
                    )
                    tmp._set_output_type(train_pairs[i][0])
                    tmp._set_n_features_in(n_features[i])
                    tmp.coef_ = coefs[i]
                    tmp.intercept_ = intercept_val
                    out_scores[i] = tmp.score(X_eval, y_eval)

        if return_models and return_scores:
            return models, out_scores
        elif return_models:
            return models
        else:
            return out_scores

    def _fit_multi_target(
        self, X_m, y_m, sample_weight_m=None, X_is_copy=False, y_is_copy=False,
    ):
        X = X_m.to_output("cupy")
        y = y_m.to_output("cupy")

        if self.fit_intercept:
            # Add column containing ones to fit intercept.
            nrow, ncol = X.shape
            X_temp = cp.empty_like(X, shape=(nrow, ncol + 1))
            X_temp[:, :ncol] = X
            X_temp[:, ncol] = 1.
            X = X_temp
            X_is_copy = True

        if sample_weight_m is not None:
            sample_weight = sample_weight_m.to_output("cupy")
            # Weights are always copied, can mutate buffer
            weight_sqrt = cp.sqrt(sample_weight, out=sample_weight)
            # Multiply by weights, reusing existing buffers when possible
            X = cp.multiply(
                X,
                weight_sqrt[:, None],
                out=X if X_is_copy or not self.copy_X else None,
            )
            y = cp.multiply(
                y,
                weight_sqrt[:, None],
                out=y if y_is_copy else None
            )

        u, s, vh = cp.linalg.svd(X, full_matrices=False)
        temp = _divide_non_zero(u.T.dot(y), s[:, None])
        coef = vh.T.dot(temp)

        if self.fit_intercept:
            intercept = CumlArray(data=coef[-1])
            coef = CumlArray(data=coef[:-1].T)
        else:
            intercept = 0.0
            coef = CumlArray(data=coef.T)

        self.coef_ = coef
        self.intercept_ = intercept

    @staticmethod
    def _more_static_tags():
        return {"multioutput": True}
