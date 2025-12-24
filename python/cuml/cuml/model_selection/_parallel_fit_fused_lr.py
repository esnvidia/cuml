# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import cupy as cp
import numpy as np


@dataclass
class _FusedLRResult:
    coef: cp.ndarray  # (n_models, n_features)
    intercept: cp.ndarray  # (n_models,)


_KERNELS = {}


def _get_kernel(dtype: np.dtype) -> cp.RawKernel:
    """Return a RawKernel that computes per-window OLS via fused normal equations.

    One block computes one window. Intended for small n_features (<=32) and moderate window sizes.
    Accumulates in double for better numerical agreement vs reference.
    """
    dt = np.dtype(dtype)
    key = str(dt)
    k = _KERNELS.get(key)
    if k is not None:
        return k

    if dt == np.float32:
        in_t = "float"
        out_t = "float"
    elif dt == np.float64:
        in_t = "double"
        out_t = "double"
    else:
        raise TypeError(f"unsupported dtype: {dt}")

    # Max features supported by this kernel (compile-time constant).
    # Keep small to limit shared memory usage (Nsight: avoid >48KB SMEM).
    max_d = 16

    # v2 (fast float32 path): row-parallel accumulation with float atomics in shared memory.
    # This matches the original high-throughput behavior observed in earlier Nsight captures.
    if dt == np.float32:
        code = r"""
        __device__ __forceinline__ double dabs(double x) { return x < 0.0 ? -x : x; }
        __device__ __forceinline__ double warp_reduce_sum(double v) {
          for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
          }
          return v;
        }

        extern "C" __global__
        void fused_lr_ne(const float* __restrict__ X,
                         long long ld,
                         const float* __restrict__ y,
                         long long n_total,
                         const long long* __restrict__ starts,
                         int n_models,
                         int window,
                         int d,
                         float* __restrict__ out_coef,
                         float* __restrict__ out_intercept)
        {
          const int m = (int)blockIdx.x;
          if (m >= n_models) return;
          const long long start0 = starts[m];

          const int tid = (int)threadIdx.x;
          const int lane = tid & 31;
          const int warp = tid >> 5;
          const int nwarps = ((int)blockDim.x) >> 5;

          // Shared storage for final stats (written by warps, read by thread 0).
          __shared__ double SX[16];
          __shared__ double SXY[16];
          __shared__ double SXX[16*16];
          __shared__ double SY;

          // Per-warp partials in registers.
          double sx = 0.0;
          double sxy = 0.0;
          double sy = 0.0;

          // Each warp owns 1 feature index j = warp (for SX/SXY) if j<d.
          const int j_feat = warp;

          // Each warp owns multiple SXX entries: idx = warp, warp+nwarps, ... up to d*d.
          // Each lane handles multiple idx values via idx = base + lane, base += 32.
          const int idx0 = warp;
          const int lane_stride = 32;
          // Max entries per lane for d<=16: ceil(256/32)=8
          double sxx_acc[8];
          #pragma unroll
          for (int t = 0; t < 8; ++t) sxx_acc[t] = 0.0;

          // Accumulate over rows.
          for (int r = lane; r < window; r += lane_stride) {
            const long long row = start0 + (long long)r;

            // Load x vector for this row: lanes < d load X[row, lane] (F-order column-major).
            double x_lane = 0.0;
            if (lane < d) {
              x_lane = (double)X[row + (long long)lane * ld];
            }

            // Load y in lane 0 (per warp); broadcast to lanes.
            double yv = 0.0;
            if (lane == 0) yv = (double)y[row];
            yv = __shfl_sync(0xffffffff, yv, 0);

            // SY: only warp 0 accumulates (to avoid redundant work).
            if (warp == 0) {
              if (lane == 0) sy += yv;
            }

            // SX/SXY: warp j_feat accumulates feature j_feat.
            if (j_feat < d) {
              // broadcast x_j_feat from lane=j_feat
              const double xj = __shfl_sync(0xffffffff, x_lane, j_feat);
              if (lane == 0) {
                sx += xj;
                sxy += xj * yv;
              }
            }

            // SXX: for each idx owned by this warp, lane computes a slice and reduces.
            int t = 0;
            for (int base = idx0; base < d * d; base += nwarps) {
              const int idx = base;
              // Each idx corresponds to one (i,j) pair. We compute full matrix.
              const int i = idx / d;
              const int j = idx - i * d;
              const double xi = __shfl_sync(0xffffffff, x_lane, i);
              const double xj = __shfl_sync(0xffffffff, x_lane, j);
              // Only one lane per warp updates each accumulator to keep it simple.
              if (lane == 0) {
                sxx_acc[t] += xi * xj;
              }
              ++t;
              if (t >= 8) break;
            }
          }

          // Reduce within warp and write to shared.
          if (warp == 0) {
            double sy_sum = 0.0;
            if (lane == 0) sy_sum = sy;
            // sy lives only in lane0; no need for warp reduction.
            if (lane == 0) SY = sy_sum;
          }

          if (j_feat < d) {
            // sx/sxy also only in lane0 of owning warp.
            if (lane == 0) {
              SX[j_feat] = sx;
              SXY[j_feat] = sxy;
            }
          }

          // Write SXX: only lane0 per warp writes its owned indices.
          if (lane == 0) {
            int t = 0;
            for (int base = idx0; base < d * d; base += nwarps) {
              const int idx = base;
              const int i = idx / d;
              const int j = idx - i * d;
              SXX[i*16 + j] = sxx_acc[t];
              ++t;
              if (t >= 8) break;
            }
          }

          __syncthreads();

          // Solve (thread 0)
          if (tid == 0) {
            const double inv_n = 1.0 / (double)window;
            double muX[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) muX[j] = 0.0;
            for (int j = 0; j < d; ++j) muX[j] = SX[j] * inv_n;
            const double muY = SY * inv_n;

            double A[16*16];
            double b[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) b[j] = 0.0;

            for (int i = 0; i < d; ++i) {
              b[i] = SXY[i] - (double)window * muX[i] * muY;
              for (int j = 0; j < d; ++j) {
                A[i*16 + j] = SXX[i*16 + j] - (double)window * muX[i] * muX[j];
              }
            }

            for (int k = 0; k < d; ++k) {
              int piv = k;
              double best = dabs(A[k*16 + k]);
              for (int r = k + 1; r < d; ++r) {
                const double v = dabs(A[r*16 + k]);
                if (v > best) { best = v; piv = r; }
              }
              if (piv != k) {
                for (int c = k; c < d; ++c) {
                  const double tmp = A[k*16 + c];
                  A[k*16 + c] = A[piv*16 + c];
                  A[piv*16 + c] = tmp;
                }
                const double tb = b[k]; b[k] = b[piv]; b[piv] = tb;
              }
              double diag = A[k*16 + k];
              if (diag == 0.0) diag = 1e-30;
              for (int r = k + 1; r < d; ++r) {
                const double f = A[r*16 + k] / diag;
                A[r*16 + k] = 0.0;
                for (int c = k + 1; c < d; ++c) {
                  A[r*16 + c] -= f * A[k*16 + c];
                }
                b[r] -= f * b[k];
              }
            }

            double wsol[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) wsol[j] = 0.0;
            for (int i = d - 1; i >= 0; --i) {
              double s = b[i];
              for (int j = i + 1; j < d; ++j) {
                s -= A[i*16 + j] * wsol[j];
              }
              double diag = A[i*16 + i];
              if (diag == 0.0) diag = 1e-30;
              wsol[i] = s / diag;
            }

            double intercept = muY;
            for (int j = 0; j < d; ++j) intercept -= muX[j] * wsol[j];
            out_intercept[m] = (float)intercept;
            for (int j = 0; j < d; ++j) out_coef[(long long)m * d + j] = (float)wsol[j];
          }
        }

        """
    else:
        code = r"""
        __device__ __forceinline__ double dabs(double x) { return x < 0.0 ? -x : x; }
        __device__ __forceinline__ double warp_reduce_sum(double v) {
          // Full warp reduction (assumes active mask = 0xffffffff)
          for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
          }
          return v;
        }

        extern "C" __global__
        void fused_lr_ne(const double* __restrict__ X,
                         long long ld,  // leading dimension (rows) for F-order X
                         const double* __restrict__ y,
                         long long n_total,
                         const long long* __restrict__ starts,
                         int n_models,
                         int window,
                         int d,
                         double* __restrict__ out_coef,      // (n_models, d) row-major
                         double* __restrict__ out_intercept) // (n_models,)
        {
          const int m = (int)blockIdx.x;
          if (m >= n_models) return;
          const long long start = starts[m];

          const int tid = (int)threadIdx.x;
          const int lane = tid & 31;
          const int warp = tid >> 5;
          const int nwarps = ((int)blockDim.x) >> 5;

          __shared__ double SX[16];
          __shared__ double SXY[16];
          __shared__ double SXX[16*16];
          __shared__ double SY;

          // Sum y (warp 0)
          if (warp == 0) {
            double local = 0.0;
            for (int r = lane; r < window; r += 32) {
              const long long row = start + (long long)r;
              local += (double)y[row];
            }
            double sum = warp_reduce_sum(local);
            if (lane == 0) SY = sum;
          }

          // Sum X and X*y (features distributed across warps)
          for (int j = warp; j < d; j += nwarps) {
            double sx = 0.0;
            double sxy = 0.0;
            for (int r = lane; r < window; r += 32) {
              const long long row = start + (long long)r;
              const double xv = (double)X[row + (long long)j * ld];
              const double yv = (double)y[row];
              sx += xv;
              sxy += xv * yv;
            }
            sx = warp_reduce_sum(sx);
            sxy = warp_reduce_sum(sxy);
            if (lane == 0) {
              SX[j] = sx;
              SXY[j] = sxy;
            }
          }

          // Sum X^T X (pairs distributed across warps)
          for (int idx = warp; idx < d * d; idx += nwarps) {
            const int i = idx / d;
            const int j = idx - i * d;
            double sxx = 0.0;
            for (int r = lane; r < window; r += 32) {
              const long long row = start + (long long)r;
              const double xi = (double)X[row + (long long)i * ld];
              const double xj = (double)X[row + (long long)j * ld];
              sxx += xi * xj;
            }
            sxx = warp_reduce_sum(sxx);
            if (lane == 0) {
              SXX[i*16 + j] = sxx;
            }
          }

          __syncthreads();

          // Centered normal equations in-place:
          //   A = Xc^T Xc = SXX - n * muX muX^T
          //   b = Xc^T yc = SXY - n * muX * muY
          const double inv_n = 1.0 / (double)window;
          const double muY = SY * inv_n;

          // Convert SX to muX in-place
          for (int j = tid; j < d; j += (int)blockDim.x) {
            SX[j] = SX[j] * inv_n;
          }
          __syncthreads();

          for (int idx = tid; idx < d * d; idx += (int)blockDim.x) {
            const int i = idx / d;
            const int j = idx - i * d;
            SXX[i*16 + j] = SXX[i*16 + j] - (double)window * SX[i] * SX[j];
          }
          for (int j = tid; j < d; j += (int)blockDim.x) {
            SXY[j] = SXY[j] - (double)window * SX[j] * muY;
          }
          __syncthreads();

          // Solve A w = b (thread 0; d<=16)
          if (tid == 0) {
            for (int k = 0; k < d; ++k) {
              int piv = k;
              double best = dabs(SXX[k*16 + k]);
              for (int r = k + 1; r < d; ++r) {
                const double v = dabs(SXX[r*16 + k]);
                if (v > best) { best = v; piv = r; }
              }
              if (piv != k) {
                for (int c = k; c < d; ++c) {
                  const double tmp = SXX[k*16 + c];
                  SXX[k*16 + c] = SXX[piv*16 + c];
                  SXX[piv*16 + c] = tmp;
                }
                const double tb = SXY[k]; SXY[k] = SXY[piv]; SXY[piv] = tb;
              }

              double diag = SXX[k*16 + k];
              if (diag == 0.0) diag = 1e-30;
              for (int r = k + 1; r < d; ++r) {
                const double f = SXX[r*16 + k] / diag;
                SXX[r*16 + k] = 0.0;
                for (int c = k + 1; c < d; ++c) {
                  SXX[r*16 + c] -= f * SXX[k*16 + c];
                }
                SXY[r] -= f * SXY[k];
              }
            }

            double wsol[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) wsol[j] = 0.0;
            for (int i = d - 1; i >= 0; --i) {
              double s = SXY[i];
              for (int j = i + 1; j < d; ++j) {
                s -= SXX[i*16 + j] * wsol[j];
              }
              double diag = SXX[i*16 + i];
              if (diag == 0.0) diag = 1e-30;
              wsol[i] = s / diag;
            }

            double intercept = muY;
            for (int j = 0; j < d; ++j) intercept -= SX[j] * wsol[j];
            out_intercept[m] = (double)intercept;
            for (int j = 0; j < d; ++j) {
              out_coef[(long long)m * d + j] = (double)wsol[j];
            }
          }
        }

    """

    k = cp.RawKernel(code, "fused_lr_ne")
    _KERNELS[key] = k
    return k


_PREFIX_KERNELS = {}

_PREFIX_SCAN_KERNELS = {}


def _get_prefix_scan_kernel(dtype: np.dtype) -> cp.RawKernel:
    """Single-kernel row-major prefix scan over rows (axis=0) for small n_total.

    This replaces cp.cumsum(vals, axis=0) to reduce launch overhead and improve locality
    for the walk-forward case (n_total~5k, n_ch~O(100)).
    """
    dt = np.dtype(dtype)
    key = str(dt)
    k = _PREFIX_SCAN_KERNELS.get(key)
    if k is not None:
        return k

    if dt == np.float32:
        t = 'float'
    elif dt == np.float64:
        t = 'double'
    else:
        raise TypeError(f'unsupported dtype: {dt}')

    code = rf"""
    extern "C" __global__
    void prefix_scan_rows(const {t}* __restrict__ vals,
                          int n_total,
                          int n_ch,
                          {t}* __restrict__ pref)
    {{
      // pref has shape (n_total+1, n_ch) row-major
      const int tid = (int)threadIdx.x;
      // Initialize row 0
      for (int c = tid; c < n_ch; c += (int)blockDim.x) {{
        pref[c] = ({t})0;
      }}
      __syncthreads();

      // Running sums per channel (kept in registers per thread for its channels)
      for (int c = tid; c < n_ch; c += (int)blockDim.x) {{
        {t} acc = ({t})0;
        for (int r = 0; r < n_total; ++r) {{
          acc += vals[(long long)r * n_ch + c];
          pref[(long long)(r + 1) * n_ch + c] = acc;
        }}
      }}
    }}
    """

    k = cp.RawKernel(code, 'prefix_scan_rows')
    _PREFIX_SCAN_KERNELS[key] = k
    return k


def _get_prefix_kernels(dtype: np.dtype) -> tuple[cp.RawKernel, cp.RawKernel]:
    """Return (build_vals, solve_from_prefix) kernels for the given dtype.

    - float32 path uses float32 prefix buffers for speed.
    - float64 path uses float64 prefix buffers.

    This is internal and validated via numerical-equivalence tests.
    """
    dt = np.dtype(dtype)
    key = f"v2_{dt}"
    k = _PREFIX_KERNELS.get(key)
    if k is not None:
        return k

    MAX_D = 16
    # MAX_CH for max_d=16: 1 + 2*d + d*(d+1)/2 = 169
    MAX_CH = 169

    if dt == np.float32:
        build_code = r"""
        extern "C" __global__
        void build_vals(const float* __restrict__ X,
                        long long ld,
                        const float* __restrict__ y,
                        int n_total,
                        int d,
                        int n_ch,
                        float* __restrict__ out_vals)
        {
          const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
          if (r >= n_total) return;

          const float yv = y[r];
          out_vals[(long long)r * n_ch + 0] = yv;

          for (int j = 0; j < d; ++j) {
            const float xj = X[(long long)r + (long long)j * ld];
            out_vals[(long long)r * n_ch + (1 + j)] = xj;
            out_vals[(long long)r * n_ch + (1 + d + j)] = xj * yv;
          }

          int k = 1 + 2 * d;
          for (int i = 0; i < d; ++i) {
            const float xi = X[(long long)r + (long long)i * ld];
            for (int j = i; j < d; ++j) {
              const float xj = X[(long long)r + (long long)j * ld];
              out_vals[(long long)r * n_ch + k] = xi * xj;
              ++k;
            }
          }
        }
        """

        solve_code = rf"""
        __device__ __forceinline__ double dabs(double x) {{ return x < 0.0 ? -x : x; }}

        extern "C" __global__
        void solve_from_prefix(const float* __restrict__ pref,
                               int n_ch,
                               const long long* __restrict__ starts,
                               int n_models,
                               int window,
                               int d,
                               float* __restrict__ out_coef,
                               float* __restrict__ out_intercept)
        {{
          const int m = (int)blockIdx.x;
          if (m >= n_models) return;

          const long long s = starts[m];
          const long long e = s + (long long)window;

          __shared__ float S[{MAX_CH}];
          for (int c = (int)threadIdx.x; c < n_ch; c += (int)blockDim.x) {{
            const float a = pref[e * (long long)n_ch + (long long)c];
            const float b = pref[s * (long long)n_ch + (long long)c];
            S[c] = a - b;
          }}
          __syncthreads();

          if (threadIdx.x == 0) {{
            const double SY = (double)S[0];
            const double inv_n = 1.0 / (double)window;
            const double muY = SY * inv_n;

            double muX[{MAX_D}];
            #pragma unroll
            for (int j = 0; j < {MAX_D}; ++j) muX[j] = 0.0;
            for (int j = 0; j < d; ++j) muX[j] = (double)S[1 + j] * inv_n;

            double A[{MAX_D}*{MAX_D}];
            double bvec[{MAX_D}];
            #pragma unroll
            for (int j = 0; j < {MAX_D}; ++j) bvec[j] = 0.0;

            for (int i = 0; i < d; ++i) {{
              const double sxy = (double)S[1 + d + i];
              bvec[i] = sxy - (double)window * muX[i] * muY;
            }}

            int k = 1 + 2 * d;
            for (int i = 0; i < d; ++i) {{
              for (int j = i; j < d; ++j) {{
                const double sxx = (double)S[k++];
                const double v = sxx - (double)window * muX[i] * muX[j];
                A[i*{MAX_D} + j] = v;
                A[j*{MAX_D} + i] = v;
              }}
            }}

            for (int kk = 0; kk < d; ++kk) {{
              int piv = kk;
              double best = dabs(A[kk*{MAX_D} + kk]);
              for (int r = kk + 1; r < d; ++r) {{
                const double v = dabs(A[r*{MAX_D} + kk]);
                if (v > best) {{ best = v; piv = r; }}
              }}
              if (piv != kk) {{
                for (int c = kk; c < d; ++c) {{
                  const double tmp = A[kk*{MAX_D} + c];
                  A[kk*{MAX_D} + c] = A[piv*{MAX_D} + c];
                  A[piv*{MAX_D} + c] = tmp;
                }}
                const double tb = bvec[kk]; bvec[kk] = bvec[piv]; bvec[piv] = tb;
              }}
              double diag = A[kk*{MAX_D} + kk];
              if (diag == 0.0) diag = 1e-30;
              for (int r = kk + 1; r < d; ++r) {{
                const double f = A[r*{MAX_D} + kk] / diag;
                A[r*{MAX_D} + kk] = 0.0;
                for (int c = kk + 1; c < d; ++c) {{
                  A[r*{MAX_D} + c] -= f * A[kk*{MAX_D} + c];
                }}
                bvec[r] -= f * bvec[kk];
              }}
            }}

            double wsol[{MAX_D}];
            #pragma unroll
            for (int j = 0; j < {MAX_D}; ++j) wsol[j] = 0.0;
            for (int i = d - 1; i >= 0; --i) {{
              double s2 = bvec[i];
              for (int j = i + 1; j < d; ++j) {{
                s2 -= A[i*{MAX_D} + j] * wsol[j];
              }}
              double diag = A[i*{MAX_D} + i];
              if (diag == 0.0) diag = 1e-30;
              wsol[i] = s2 / diag;
            }}

            double intercept = muY;
            for (int j = 0; j < d; ++j) intercept -= muX[j] * wsol[j];
            out_intercept[m] = (float)intercept;
            for (int j = 0; j < d; ++j) out_coef[(long long)m * d + j] = (float)wsol[j];
          }}
        }}
        """

        build_k = cp.RawKernel(build_code, "build_vals")
        solve_k = cp.RawKernel(solve_code, "solve_from_prefix")
        _PREFIX_KERNELS[key] = (build_k, solve_k)
        return build_k, solve_k

    if dt != np.float64:
        raise TypeError(f"unsupported dtype for prefix backend: {dt}")

    build_code = r"""
    extern "C" __global__
    void build_vals(const double* __restrict__ X,
                    long long ld,
                    const double* __restrict__ y,
                    int n_total,
                    int d,
                    int n_ch,
                    double* __restrict__ out_vals)
    {
      const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
      if (r >= n_total) return;

      const double yv = y[r];
      out_vals[(long long)r * n_ch + 0] = yv;

      for (int j = 0; j < d; ++j) {
        const double xj = X[(long long)r + (long long)j * ld];
        out_vals[(long long)r * n_ch + (1 + j)] = xj;
        out_vals[(long long)r * n_ch + (1 + d + j)] = xj * yv;
      }

      int k = 1 + 2 * d;
      for (int i = 0; i < d; ++i) {
        const double xi = X[(long long)r + (long long)i * ld];
        for (int j = i; j < d; ++j) {
          const double xj = X[(long long)r + (long long)j * ld];
          out_vals[(long long)r * n_ch + k] = xi * xj;
          ++k;
        }
      }
    }
    """

    solve_code = rf"""
    __device__ __forceinline__ double dabs(double x) {{ return x < 0.0 ? -x : x; }}

    extern "C" __global__
    void solve_from_prefix(const double* __restrict__ pref,
                           int n_ch,
                           const long long* __restrict__ starts,
                           int n_models,
                           int window,
                           int d,
                           double* __restrict__ out_coef,
                           double* __restrict__ out_intercept)
    {{
      const int m = (int)blockIdx.x;
      if (m >= n_models) return;

      const long long s = starts[m];
      const long long e = s + (long long)window;

      __shared__ double S[{MAX_CH}];
      for (int c = (int)threadIdx.x; c < n_ch; c += (int)blockDim.x) {{
        const double a = pref[e * (long long)n_ch + (long long)c];
        const double b = pref[s * (long long)n_ch + (long long)c];
        S[c] = a - b;
      }}
      __syncthreads();

      if (threadIdx.x == 0) {{
        const double SY = S[0];
        const double inv_n = 1.0 / (double)window;
        const double muY = SY * inv_n;

        double muX[{MAX_D}];
        #pragma unroll
        for (int j = 0; j < {MAX_D}; ++j) muX[j] = 0.0;
        for (int j = 0; j < d; ++j) muX[j] = S[1 + j] * inv_n;

        double A[{MAX_D}*{MAX_D}];
        double bvec[{MAX_D}];
        #pragma unroll
        for (int j = 0; j < {MAX_D}; ++j) bvec[j] = 0.0;

        for (int i = 0; i < d; ++i) {{
          const double sxy = S[1 + d + i];
          bvec[i] = sxy - (double)window * muX[i] * muY;
        }}

        int k = 1 + 2 * d;
        for (int i = 0; i < d; ++i) {{
          for (int j = i; j < d; ++j) {{
            const double sxx = S[k++];
            const double v = sxx - (double)window * muX[i] * muX[j];
            A[i*{MAX_D} + j] = v;
            A[j*{MAX_D} + i] = v;
          }}
        }}

        for (int kk = 0; kk < d; ++kk) {{
          int piv = kk;
          double best = dabs(A[kk*{MAX_D} + kk]);
          for (int r = kk + 1; r < d; ++r) {{
            const double v = dabs(A[r*{MAX_D} + kk]);
            if (v > best) {{ best = v; piv = r; }}
          }}
          if (piv != kk) {{
            for (int c = kk; c < d; ++c) {{
              const double tmp = A[kk*{MAX_D} + c];
              A[kk*{MAX_D} + c] = A[piv*{MAX_D} + c];
              A[piv*{MAX_D} + c] = tmp;
            }}
            const double tb = bvec[kk]; bvec[kk] = bvec[piv]; bvec[piv] = tb;
          }}
          double diag = A[kk*{MAX_D} + kk];
          if (diag == 0.0) diag = 1e-30;
          for (int r = kk + 1; r < d; ++r) {{
            const double f = A[r*{MAX_D} + kk] / diag;
            A[r*{MAX_D} + kk] = 0.0;
            for (int c = kk + 1; c < d; ++c) {{
              A[r*{MAX_D} + c] -= f * A[kk*{MAX_D} + c];
            }}
            bvec[r] -= f * bvec[kk];
          }}
        }}

        double wsol[{MAX_D}];
        #pragma unroll
        for (int j = 0; j < {MAX_D}; ++j) wsol[j] = 0.0;
        for (int i = d - 1; i >= 0; --i) {{
          double s2 = bvec[i];
          for (int j = i + 1; j < d; ++j) {{
            s2 -= A[i*{MAX_D} + j] * wsol[j];
          }}
          double diag = A[i*{MAX_D} + i];
          if (diag == 0.0) diag = 1e-30;
          wsol[i] = s2 / diag;
        }}

        double intercept = muY;
        for (int j = 0; j < d; ++j) intercept -= muX[j] * wsol[j];
        out_intercept[m] = intercept;
        for (int j = 0; j < d; ++j) out_coef[(long long)m * d + j] = wsol[j];
      }}
    }}
    """

    build_k = cp.RawKernel(build_code, "build_vals")
    solve_k = cp.RawKernel(solve_code, "solve_from_prefix")
    _PREFIX_KERNELS[key] = (build_k, solve_k)
    return build_k, solve_k


def _fused_lr_prefix_many_slices(
    X: cp.ndarray,
    y: cp.ndarray,
    starts: cp.ndarray,
    *,
    window: int,
    n_features: int,
) -> _FusedLRResult:
    """Prefix-sums backend for overlapping slice windows.

    This backend is designed for highly-overlapping windows (walk-forward). It builds a
    compact set of cumulative statistics once and then solves one small dxd system per
    window.
    """
    n_total, d = X.shape
    if d != n_features:
        raise ValueError("n_features must match X.shape[1]")

    dt = np.dtype(X.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"unsupported dtype: {dt}")

    # Channel layout: y, x(d), x*y(d), x*x upper-tri (d*(d+1)//2)
    n_ch = int(1 + 2 * d + (d * (d + 1)) // 2)

    Xp = cp.asarray(X, dtype=dt, order="F")
    yp = cp.asarray(y.reshape(-1), dtype=dt)

    vals = cp.empty((n_total, n_ch), dtype=dt)
    build_k, solve_k = _get_prefix_kernels(dt)

    threads = 256
    blocks = (n_total + threads - 1) // threads
    build_k(
        (blocks,),
        (threads,),
        (
            Xp,
            np.int64(Xp.shape[0]),
            yp,
            np.int32(n_total),
            np.int32(d),
            np.int32(n_ch),
            vals,
        ),
    )

    pref = cp.empty((n_total + 1, n_ch), dtype=dt)
    pref[0, :] = dt.type(0.0)
    # Use dtype-matched cumsum for speed on float32.
    scan_k = _get_prefix_scan_kernel(dt)
    # Single-block scan is sufficient for n_total ~ few thousand and keeps overhead minimal.
    scan_k((1,), (128,), (vals, np.int32(n_total), np.int32(n_ch), pref))

    out_coef = cp.empty((int(starts.size), d), dtype=dt)
    out_intercept = cp.empty((int(starts.size),), dtype=dt)

    solve_threads = 32
    solve_k(
        (int(starts.size),),
        (solve_threads,),
        (
            pref,
            np.int32(n_ch),
            starts,
            np.int32(int(starts.size)),
            np.int32(window),
            np.int32(d),
            out_coef,
            out_intercept,
        ),
    )

    return _FusedLRResult(coef=out_coef, intercept=out_intercept)


def fused_linear_regression_many_slices(
    X: cp.ndarray,
    y: cp.ndarray,
    starts: cp.ndarray,
    *,
    window: int,
    n_features: int,
) -> _FusedLRResult:
    """Compute OLS fits for many slice windows using fused normal equations.

    Parameters
    ----------
    X : cupy.ndarray
        Full design matrix, expected F-order for best performance.
    y : cupy.ndarray
        Full target vector, shape (n_total,).
    starts : cupy.ndarray[int64]
        Start index for each window; each window is [start, start+window).
    window : int
        Number of rows per train window.
    n_features : int
        Number of columns/features.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        y = y.reshape(-1)
    if starts.dtype != cp.int64:
        starts = starts.astype(cp.int64)
    if starts.ndim != 1:
        starts = starts.reshape(-1)
    if window < 2:
        raise ValueError("window must be >= 2")
    if n_features < 1 or n_features > 16:
        raise ValueError("n_features must be in [1, 16] for fused backend")

    n_total, d = X.shape
    if d != n_features:
        raise ValueError("n_features must match X.shape[1]")
    if int(starts.size) == 0:
        return _FusedLRResult(
            coef=cp.empty((0, d), dtype=X.dtype), intercept=cp.empty((0,), dtype=X.dtype)
        )

    # Optional bounds check. By default we skip this to avoid host synchronization
    # on the fast path. Enable for debugging with:
    #   CUML_PARALLEL_FIT_FUSED_LR_VALIDATE_BOUNDS=1
    validate_bounds = os.environ.get("CUML_PARALLEL_FIT_FUSED_LR_VALIDATE_BOUNDS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if validate_bounds:
        max_start = int(cp.max(starts).get())
        if max_start + window > n_total:
            raise ValueError("window exceeds X/y length for some starts")

    dtype = X.dtype
    if y.dtype != dtype:
        y = y.astype(dtype, copy=False)

    out_coef = cp.empty((int(starts.size), d), dtype=dtype)
    out_intercept = cp.empty((int(starts.size),), dtype=dtype)

    use_prefix = os.environ.get("CUML_PARALLEL_FIT_FUSED_LR_USE_PREFIX", "1").lower() in (
        "1", "true", "yes", "on"
    )
    if use_prefix:
        return _fused_lr_prefix_many_slices(X, y, starts, window=window, n_features=n_features)

    k = _get_kernel(np.dtype(dtype))
    threads = 128
    blocks = int(starts.size)
    k(
        (blocks,),
        (threads,),
        (
            X,
            np.int64(X.shape[0]),  # ld
            y,
            np.int64(n_total),
            starts,
            np.int32(int(starts.size)),
            np.int32(window),
            np.int32(d),
            out_coef,
            out_intercept,
        ),
    )

    return _FusedLRResult(coef=out_coef, intercept=out_intercept)


