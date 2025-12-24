# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pytest
import os

from cuml.linear_model import LinearRegression
from cuml.model_selection import parallel_fit
from cuml.internals.array import CumlArray


class _SimpleWalkForward:
    """Minimal splitter compatible with cuml.model_selection.parallel_fit."""

    def __init__(self, train_window: int, val_size: int, step_size: int = 1):
        self.train_window = train_window
        self.val_size = val_size
        self.step_size = step_size

    def split(self, X, y=None):
        n = len(X)
        for start in range(0, n - self.train_window - self.val_size + 1, self.step_size):
            # Use slice windows to avoid per-split gather indices (fast path).
            train_idx = slice(start, start + self.train_window)
            val_idx = slice(start + self.train_window, start + self.train_window + self.val_size)
            yield train_idx, val_idx


def _to_cupy(x):
    return x.to_output("cupy") if hasattr(x, "to_output") else cp.asarray(x)


class _WalkForwardSlices:
    """Walk-forward splitter yielding `slice` windows (fast path, avoids gather indexing)."""

    def __init__(
        self,
        train_window: int,
        val_size: int,
        step_size: int = 1,
        max_splits: int | None = None,
    ):
        self.train_window = train_window
        self.val_size = val_size
        self.step_size = step_size
        self.max_splits = max_splits

    def split(self, X, y=None):
        n = len(X)
        emitted = 0
        for start in range(0, n - self.train_window - self.val_size + 1, self.step_size):
            tr = slice(start, start + self.train_window)
            va = slice(start + self.train_window, start + self.train_window + self.val_size)
            yield tr, va
            emitted += 1
            if self.max_splits is not None and emitted >= self.max_splits:
                return


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_parallel_fit_accepts_cumlarray(dtype):
    n = 64
    d = 3
    X = cp.arange(n * d, dtype=dtype).reshape(n, d, order="F")
    w = cp.arange(1, d + 1, dtype=dtype)
    y = X @ w + dtype(1.0)

    X_ca = CumlArray.from_input(X, order="F")
    y_ca = CumlArray.from_input(y)

    res = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X_ca,
        y=y_ca,
        splitter=_SimpleWalkForward(train_window=20, val_size=1, step_size=7),
        max_concurrency=3,
        return_models=False,
        return_scores=True,
        scoring="r2",
        return_coef=True,
        return_intercept=True,
        score_on="val",
    )

    assert res.scores is not None
    assert res.coef_ is not None
    assert res.intercept_ is not None
    assert int(res.scores.shape[0]) > 0


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_parallel_fit_accepts_cudf(dtype):
    cudf = pytest.importorskip("cudf")

    n = 80
    d = 4
    X = cp.arange(n * d, dtype=dtype).reshape(n, d, order="F")
    w = cp.arange(1, d + 1, dtype=dtype)
    y = X @ w + dtype(2.0)

    X_df = cudf.DataFrame({f"c{i}": X[:, i] for i in range(d)})
    y_s = cudf.Series(y)

    res = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X_df,
        y=y_s,
        splitter=_SimpleWalkForward(train_window=25, val_size=1, step_size=6),
        max_concurrency=4,
        return_models=False,
        return_scores=True,
        scoring="r2",
        return_coef=True,
        return_intercept=True,
        score_on="val",
    )

    assert res.scores is not None
    assert res.coef_ is not None
    assert res.intercept_ is not None
    assert int(res.scores.shape[0]) > 0


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_parallel_fit_matches_sequential(dtype):
    # Build an easy linear relationship: y = X @ w + b
    n = 128
    d = 5
    X = cp.arange(n * d, dtype=dtype).reshape(n, d, order="F")
    w = cp.arange(1, d + 1, dtype=dtype)
    b = dtype(3.0)
    y = X @ w + b

    splitter = _SimpleWalkForward(train_window=50, val_size=1, step_size=3)

    # Sequential baseline
    seq_models = []
    seq_scores = []
    for tr, va in splitter.split(X, y):
        m = LinearRegression(fit_intercept=True, algorithm="eig", copy_X=True)
        m.fit(X[tr], y[tr])
        seq_models.append(m)
        seq_scores.append(float(m.score(X[va], y[va])))

    # Parallel fit
    res = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X,
        y=y,
        splitter=_SimpleWalkForward(train_window=50, val_size=1, step_size=3),
        max_concurrency=4,
        return_models=True,
        return_scores=True,
        scoring="r2",
        return_coef=True,
        return_intercept=True,
        score_on="val",
    )

    assert res.models is not None
    assert len(res.models) == len(seq_models)
    assert res.scores is not None
    assert res.coef_ is not None
    assert res.intercept_ is not None

    # Compare coefficients/intercepts per split
    for i, (m_seq, m_par) in enumerate(zip(seq_models, res.models)):
        cp.testing.assert_allclose(_to_cupy(m_par.coef_), _to_cupy(m_seq.coef_), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(float(m_par.intercept_), float(m_seq.intercept_), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(float(res.scores[i]), float(seq_scores[i]), rtol=1e-3, atol=1e-3)


def test_parallel_fit_does_not_mutate_inputs():
    X = cp.asarray(np.random.RandomState(0).randn(100, 4), dtype=cp.float32, order="F")
    y = cp.asarray(np.random.RandomState(1).randn(100), dtype=cp.float32)
    X0 = X.copy()
    y0 = y.copy()

    splitter = _SimpleWalkForward(train_window=20, val_size=1, step_size=5)
    _ = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X,
        y=y,
        splitter=splitter,
        max_concurrency=3,
        return_models=False,
        return_scores=False,
    )

    cp.testing.assert_allclose(X, X0)
    cp.testing.assert_allclose(y, y0)


@pytest.mark.parametrize("algo", ["qr", "svd"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_parallel_fit_matches_sequential_for_loop_qr_svd(dtype, algo):
    # Use a well-conditioned random problem to avoid rank-deficiency causing NaNs.
    n = 512
    d = 8
    X = cp.random.default_rng(0).standard_normal((n, d), dtype=dtype)
    X = cp.asfortranarray(X)
    w = cp.random.default_rng(1).standard_normal((d,), dtype=dtype)
    b = dtype(3.0)
    y = (X @ w + b).reshape(-1)

    splitter = _WalkForwardSlices(train_window=200, val_size=16, step_size=37)

    # Sequential baseline
    seq_models = []
    seq_scores = []
    for tr, va in splitter.split(X, y):
        m = LinearRegression(fit_intercept=True, algorithm=algo, copy_X=True)
        m.fit(X[tr], y[tr])
        seq_models.append(m)
        seq_scores.append(float(m.score(X[va], y[va])))

    # parallel_fit
    res = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm=algo, copy_X=True, handle=handle
        ),
        X=X,
        y=y,
        splitter=splitter,
        max_concurrency=4,
        return_models=True,
        return_scores=True,
        scoring="r2",
        return_coef=False,
        return_intercept=False,
        score_on="val",
    )

    assert res.models is not None
    assert res.scores is not None
    assert len(res.models) == len(seq_models)
    assert int(res.scores.shape[0]) == len(seq_scores)

    # For QR/SVD we require numerical identity to the sequential for-loop for the same backend.
    # Start strict; relax tolerances only if we observe nondeterminism in cuSOLVER.
    rtol = 1e-6 if dtype == cp.float64 else 1e-4
    atol = 1e-6 if dtype == cp.float64 else 1e-4
    for i, (m_seq, m_par) in enumerate(zip(seq_models, res.models)):
        cp.testing.assert_allclose(_to_cupy(m_par.coef_), _to_cupy(m_seq.coef_), rtol=rtol, atol=atol)
        # intercept_ is a scalar (python float)
        assert abs(float(m_par.intercept_) - float(m_seq.intercept_)) <= max(atol, rtol * abs(float(m_seq.intercept_)))
        assert abs(float(res.scores[i]) - float(seq_scores[i])) <= max(atol, rtol * abs(float(seq_scores[i])))



@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_parallel_fit_bitwise_matches_sequential_for_loop(dtype):
    # Deterministic linear relationship (should be exactly representable across both executions)
    n = 512
    d = 8
    X = cp.arange(n * d, dtype=dtype).reshape(n, d, order="F")
    w = cp.arange(1, d + 1, dtype=dtype)
    b = dtype(3.0)
    y = X @ w + b

    splitter = _WalkForwardSlices(train_window=200, val_size=16, step_size=37)

    # Sequential baseline
    seq_models = []
    seq_scores = []
    for tr, va in splitter.split(X, y):
        m = LinearRegression(fit_intercept=True, algorithm="eig", copy_X=True)
        m.fit(X[tr], y[tr])
        seq_models.append(m)
        # Use val scores to match parallel_fit configuration
        seq_scores.append(m.score(X[va], y[va]))

    # parallel_fit
    res = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X,
        y=y,
        splitter=splitter,
        max_concurrency=4,
        return_models=True,
        return_scores=True,
        scoring="r2",
        return_coef=False,
        return_intercept=False,
        score_on="val",
    )

    assert res.models is not None
    assert res.scores is not None
    assert len(res.models) == len(seq_models)
    assert int(res.scores.shape[0]) == len(seq_scores)

    # Bit-for-bit equality (exact)
    for i, (m_seq, m_par) in enumerate(zip(seq_models, res.models)):
        cp.testing.assert_array_equal(_to_cupy(m_par.coef_), _to_cupy(m_seq.coef_))
        assert float(m_par.intercept_) == float(m_seq.intercept_)
        assert float(res.scores[i]) == float(seq_scores[i])


def test_parallel_fit_beefy_walkforward_slice_fastpath_matches_sequential_subset():
    # Beefier scenario:
    # - 10 columns
    # - 5000 total rows
    # - train window 500, validation 1
    # - roll forward 1 row at a time (step=1)
    #
    # Full run yields 4500 splits; for unit tests we cap splits to keep runtime bounded.
    n_total = 5000
    d = 10
    train_window = 500
    val_size = 1
    step_size = 1
    # Cap splits in unit tests; can be overridden via env var for stress runs.
    max_splits = int(
        np.int64(
            int(os.environ.get("CUML_PARALLEL_FIT_BEEFY_MAX_SPLITS", "128"))
        )
    )

    X = cp.arange(n_total * d, dtype=cp.float32).reshape(n_total, d, order="F")
    w = cp.arange(1, d + 1, dtype=cp.float32)
    y = X @ w + cp.float32(3.0)

    splitter = _WalkForwardSlices(
        train_window=train_window,
        val_size=val_size,
        step_size=step_size,
        max_splits=max_splits,
    )

    # Sequential baseline (same windows)
    seq_models = []
    for tr, _va in splitter.split(X, y):
        m = LinearRegression(fit_intercept=True, algorithm="eig", copy_X=True)
        m.fit(X[tr], y[tr])
        seq_models.append(m)

    # Parallel fit (slice fast-path should be exercised)
    res = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X,
        y=y,
        splitter=splitter,
        max_concurrency=8,
        return_models=True,
        return_scores=False,
    )

    assert res.models is not None
    assert len(res.models) == len(seq_models)
    for m_seq, m_par in zip(seq_models, res.models):
        # For this larger scenario we still enforce bitwise coef equality.
        cp.testing.assert_array_equal(_to_cupy(m_par.coef_), _to_cupy(m_seq.coef_))
        assert float(m_par.intercept_) == float(m_seq.intercept_)


def test_parallel_fit_beefy_fused_backend_numerically_equivalent_subset(monkeypatch):
    # Ensure fused backend is enabled (it is default-on, but keep the test robust).
    monkeypatch.delenv("CUML_PARALLEL_FIT_DISABLE_FUSED_LR", raising=False)

    n_total = 5000
    d = 10
    train_window = 500
    val_size = 1
    step_size = 1
    max_splits = 128

    X = cp.random.default_rng(0).standard_normal((n_total, d), dtype=cp.float32)
    X = cp.asfortranarray(X)
    w = cp.random.default_rng(1).standard_normal((d,), dtype=cp.float32)
    y = X @ w + cp.float32(0.25)

    splitter = _WalkForwardSlices(
        train_window=train_window,
        val_size=val_size,
        step_size=step_size,
        max_splits=max_splits,
    )

    # Sequential baseline (reference)
    seq_coefs = []
    seq_intercepts = []
    for tr, _va in splitter.split(X, y):
        m = LinearRegression(fit_intercept=True, algorithm="eig", copy_X=True)
        m.fit(X[tr], y[tr])
        seq_coefs.append(_to_cupy(m.coef_))
        seq_intercepts.append(float(m.intercept_))

    seq_coefs = cp.stack(seq_coefs, axis=0)
    seq_intercepts = cp.asarray(seq_intercepts, dtype=cp.float32)

    # Fused backend: throughput-only mode (no models returned)
    _ = parallel_fit(
        estimator_factory=lambda handle=None: LinearRegression(
            fit_intercept=True, algorithm="eig", copy_X=True, handle=handle
        ),
        X=X,
        y=y,
        splitter=splitter,
        max_concurrency=8,
        return_models=False,
        return_scores=False,
        return_coef=False,
        return_intercept=False,
        score_on="val",
    )

    # Directly call the fused kernel to fetch outputs for comparison.
    from cuml.model_selection._parallel_fit_fused_lr import fused_linear_regression_many_slices

    starts = []
    for tr, _va in splitter.split(X, y):
        starts.append(int(tr.start))
    out = fused_linear_regression_many_slices(
        X, y.reshape(-1), cp.asarray(starts, dtype=cp.int64), window=train_window, n_features=d
    )

    # Numerical equivalence: tolerance check (not bitwise).
    cp.testing.assert_allclose(out.coef, seq_coefs, rtol=5e-4, atol=5e-4)
    cp.testing.assert_allclose(out.intercept, seq_intercepts, rtol=5e-4, atol=5e-4)


