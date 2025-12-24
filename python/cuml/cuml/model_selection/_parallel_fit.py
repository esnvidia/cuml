# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Iterable, List, Optional, Tuple

import cupy as cp
import numpy as np


@dataclass
class ParallelFitResult:
    models: Optional[List[Any]] = None
    scores: Optional[cp.ndarray] = None
    coef_: Optional[cp.ndarray] = None
    intercept_: Optional[cp.ndarray] = None


def _to_cupy_array(x) -> cp.ndarray:
    """Convert common cuML outputs to a cupy array without assuming CumlArray APIs."""
    # Many cuML attributes are CumlArray-like (have .to_output), but depending on
    # output_type they may already be cupy/numpy arrays.
    if hasattr(x, "to_output"):
        return x.to_output("cupy")
    return cp.asarray(x)


def _to_host_indices(idx):
    """Convert common GPU/CPU index containers into something cudf .iloc accepts."""
    if isinstance(idx, slice):
        return idx
    if isinstance(idx, cp.ndarray):
        return cp.asnumpy(idx)
    if hasattr(idx, "to_output"):
        return cp.asnumpy(idx.to_output("cupy"))
    if isinstance(idx, np.ndarray):
        return idx
    return idx


def _take_rows(X, idx):
    """Row selection that works for cupy/numpy/CumlArray and cudf DataFrame/Series."""
    if hasattr(X, "iloc"):
        return X.iloc[_to_host_indices(idx)]
    return X[idx]


def _maybe_enable_rmm_pool() -> None:
    """Best-effort: enable an RMM pool to avoid cudaMalloc/cudaFree churn.

    This is intentionally internal and controlled via env vars (no public API knob):
    - CUML_PARALLEL_FIT_ENABLE_RMM_POOL (default: "1")
    - CUML_PARALLEL_FIT_RMM_POOL_SIZE_BYTES (optional)
    - CUML_PARALLEL_FIT_RMM_ALLOCATOR (optional: "pool" or "arena"; default: "pool")
    """
    if os.environ.get("CUML_PARALLEL_FIT_ENABLE_RMM_POOL", "1").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    try:
        import rmm
    except Exception:
        return

    try:
        mr = rmm.mr.get_current_device_resource()
        mr_name = type(mr).__name__
        # If a pool/arena is already configured, do nothing.
        if mr_name in ("PoolMemoryResource", "ArenaMemoryResource"):
            return
        # Common wrapping adaptors; try to peek upstream if present.
        upstream = getattr(mr, "upstream_mr", None)
        if upstream is not None:
            upstream_name = type(upstream).__name__
            if upstream_name in ("PoolMemoryResource", "ArenaMemoryResource"):
                return

        if mr_name != "CudaMemoryResource":
            # Unknown MR type; don't override.
            return

        # Choose an initial size large enough to avoid frequent upstream cudaMalloc during
        # many small allocations (walk-forward / many fits). This is best-effort only.
        size_env = os.environ.get("CUML_PARALLEL_FIT_RMM_POOL_SIZE_BYTES")
        if size_env is not None:
            initial_pool_size = int(size_env)
        else:
            free, _total = cp.cuda.runtime.memGetInfo()
            # Default: min(4 GiB, 3/4 of free mem), but not less than 512 MiB.
            initial_pool_size = int(min(4 << 30, (free * 3) // 4))
            initial_pool_size = max(initial_pool_size, 512 << 20)
        # Allow the pool to grow up to a high-water mark to reduce upstream churn under fragmentation.
        try:
            free_now, _ = cp.cuda.runtime.memGetInfo()
            maximum_pool_size = int(min(8 << 30, (free_now * 7) // 8))
            maximum_pool_size = max(maximum_pool_size, initial_pool_size)
        except Exception:
            maximum_pool_size = None

        alloc_kind = os.environ.get("CUML_PARALLEL_FIT_RMM_ALLOCATOR", "").strip().lower()
        if alloc_kind not in ("arena", "pool", ""):
            alloc_kind = ""

        use_arena = (alloc_kind == "arena") and hasattr(rmm.mr, "ArenaMemoryResource")

        if use_arena:
            rmm.mr.set_current_device_resource(
                rmm.mr.ArenaMemoryResource(
                    rmm.mr.CudaMemoryResource(), arena_size=initial_pool_size
                )
            )
        else:
            rmm.mr.set_current_device_resource(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.CudaMemoryResource(),
                    initial_pool_size=initial_pool_size,
                    maximum_pool_size=maximum_pool_size,
                )
            )

        # Also route CuPy allocations through RMM when possible. This helps when kernels
        # (or libraries) allocate temporary device buffers via CuPy utilities.
        try:
            import rmm.allocators.cupy

            cp.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
        except Exception:
            pass
    except Exception:
        # Best-effort only; never fail the API.
        return


def parallel_fit(
    *,
    estimator_factory: Callable[..., Any],
    X,
    y,
    splitter,
    max_concurrency: int = 8,
    return_models: bool = True,
    return_scores: bool = False,
    scoring: str = "r2",
    return_coef: bool = False,
    return_intercept: bool = False,
    score_on: str = "val",
) -> ParallelFitResult:
    """Fit many models concurrently from a splitter (single GPU, multi-stream).

    This is designed for rolling / walk-forward style CV where creating a full
    list of (X_train, y_train) pairs upfront can be expensive.

    Parameters
    ----------
    estimator_factory
        Callable that returns an estimator instance. It must accept a `handle=`
        kwarg (cuML estimators do).
    X, y
        Full dataset.
    splitter
        Any object exposing `split(X, y=None)` yielding (train_idx, val_idx).
    max_concurrency
        Max number of CUDA streams/handles to use.
    return_models
        Return fitted estimator objects.
    return_scores
        Return scores per split (currently only `scoring='r2'` is supported).
    scoring
        Currently only 'r2' (matches cuML RegressorMixin.score).
    return_coef, return_intercept
        Return stacked coefficients / intercepts.
    score_on
        'val' or 'train' for where to compute scores if enabled.
    """
    if scoring != "r2":
        raise NotImplementedError("Only scoring='r2' is supported for now")
    if score_on not in ("val", "train"):
        raise ValueError("score_on must be either 'val' or 'train'")

    _maybe_enable_rmm_pool()

    # Fused backend (internal): default ON for the supported subset, can be disabled via env var.
    # This is intentionally separate from the reference multi-stream implementation and is not
    # bit-for-bit identical.
    disable_fused = os.environ.get("CUML_PARALLEL_FIT_DISABLE_FUSED_LR", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    use_fused = not disable_fused
    try:
        from cuml.linear_model import LinearRegression  # local import
        from ._parallel_fit_fused_lr import fused_linear_regression_many_slices
    except Exception:
        use_fused = False

    # Collect results incrementally to avoid materializing all windows at once.
    models: List[Any] = []
    scores_list: List[float] = []
    coefs_list: List[cp.ndarray] = []
    intercepts_list: List[float] = []

    batch_train: List[Tuple[Any, Any]] = []
    batch_eval: List[Tuple[Any, Any]] = []

    # Create a single estimator instance and reuse it across batches. This avoids repeatedly
    # creating/destroying CUDA streams/handles and helps internal caches (workspace reuse).
    est0 = estimator_factory(handle=None)

    # Fused backend is only supported for LinearRegression + slice splitter + no scores.
    if use_fused and isinstance(est0, LinearRegression):
        if return_models or return_scores:
            use_fused = False
        if score_on != "val":
            use_fused = False
        # Keep algorithm semantics: the fused normal-equations backend currently corresponds
        # to the OLS "eig" path only. For other algorithms, fall back to the reference
        # multi-stream fit_many implementation.
        if getattr(est0, "algorithm", "eig") != "eig":
            use_fused = False

    if use_fused and isinstance(est0, LinearRegression):
        # Only supports slice train windows with step 1 and non-cuDF inputs.
        if hasattr(X, "iloc") or hasattr(y, "iloc"):
            use_fused = False

    if use_fused and isinstance(est0, LinearRegression):
        # Collect all starts from slice splitter and run one fused kernel.
        starts = []
        window = None
        for train_idx, _val_idx in splitter.split(X, y):
            if not isinstance(train_idx, slice) or train_idx.step not in (None, 1):
                use_fused = False
                break
            s = 0 if train_idx.start is None else int(train_idx.start)
            e = int(train_idx.stop)
            wlen = e - s
            if window is None:
                window = wlen
            elif wlen != window:
                use_fused = False
                break
            starts.append(s)

        if use_fused and window is not None and len(starts) > 0:
            X_cp = X.to_output("cupy") if hasattr(X, "to_output") else cp.asarray(X)
            y_cp = y.to_output("cupy") if hasattr(y, "to_output") else cp.asarray(y)
            y_cp = y_cp.reshape(-1)
            starts_d = cp.asarray(starts, dtype=cp.int64)
            fused_out = fused_linear_regression_many_slices(
                X_cp, y_cp, starts_d, window=window, n_features=int(X_cp.shape[1])
            )
            out = ParallelFitResult()
            if return_coef:
                out.coef_ = fused_out.coef
            if return_intercept:
                out.intercept_ = fused_out.intercept
            return out

    def _flush_batch():
        nonlocal batch_train, batch_eval
        if not batch_train:
            return

        if return_scores:
            fitted_models, batch_scores = est0.fit_many(
                batch_train,
                max_concurrency=max_concurrency,
                eval_pairs=batch_eval,
                scoring="r2",
                return_models=True,
                return_scores=True,
            )
            scores_list.extend(cp.asarray(batch_scores).tolist())
        else:
            fitted_models = est0.fit_many(
                batch_train,
                max_concurrency=max_concurrency,
                return_models=True,
                return_scores=False,
            )

        if return_models:
            models.extend(fitted_models)

        if return_coef:
            for m in fitted_models:
                coefs_list.append(_to_cupy_array(m.coef_))

        if return_intercept:
            for m in fitted_models:
                intercepts_list.append(float(m.intercept_))

        batch_train = []
        batch_eval = []

    for train_idx, val_idx in splitter.split(X, y):
        # Slice fast-path (train only): avoid materializing X[tr]/y[tr] per window.
        # We only enable this for simple Python slices and non-cuDF inputs, and rely on
        # LinearRegression.fit_many to pack windows efficiently into scratch.
        use_slice_fastpath = (
            isinstance(train_idx, slice)
            and train_idx.step in (None, 1)
            and not hasattr(X, "iloc")
            and not hasattr(y, "iloc")
        )
        if use_slice_fastpath:
            X_tr = (X, y, train_idx)
            y_tr = None  # unused placeholder; fit_many will use y embedded in X_tr tuple
        else:
            X_tr = _take_rows(X, train_idx)
            y_tr = _take_rows(y, train_idx)

        if score_on == "train":
            X_ev, y_ev = X_tr, y_tr
        else:
            X_ev = _take_rows(X, val_idx)
            y_ev = _take_rows(y, val_idx)

        if use_slice_fastpath:
            batch_train.append(X_tr)  # (X_base, y_base, slice)
        else:
            batch_train.append((X_tr, y_tr))
        if return_scores:
            batch_eval.append((X_ev, y_ev))

        if len(batch_train) >= max_concurrency:
            _flush_batch()

    _flush_batch()

    out = ParallelFitResult()
    if return_models:
        out.models = models
    if return_scores:
        out.scores = cp.asarray(scores_list, dtype=cp.float64)
    if return_coef:
        out.coef_ = cp.stack(coefs_list, axis=0) if coefs_list else cp.empty((0, 0))
    if return_intercept:
        out.intercept_ = cp.asarray(intercepts_list, dtype=cp.float64)

    return out


