#!/usr/bin/env python3
"""Generate the shared Phi_2 table used by sparse-history GPIS.

Requires numpy/scipy. Run from any directory; the default output is relative
to the repository. The binary contains little-endian float32 probability
nodes followed by per-cell absolute interpolation-error estimates. Both
arrays exploit a/b symmetry. Cubic interpolation uses theta=asin(rho).

The interpolation part of the error is bounded using fourth derivatives on
the complete four-node stencil hull and cubic Lebesgue constants. Numerical
node errors use independently ordered Gauss-Legendre rules, a conservative
absolute roundoff allowance, and exact float32 quantization error. Thus the
reported total is a numerical error estimate, not an interval-arithmetic
certificate. Runtime callers still enforce their relative error budget and
fall back to numerical integration for rare tails and ill-conditioned cells.
"""

import argparse
import itertools
from pathlib import Path
import struct
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr


INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def phi(x):
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def product_interval(al, ah, bl, bh):
    v = np.array(np.broadcast_arrays(al * bl, al * bh, ah * bl, ah * bh))
    return np.min(v, axis=0), np.max(v, axis=0)


def power_phi_max(lo, hi, power):
    def f(x):
        return np.abs(x) ** power * phi(x)
    result = np.maximum(f(lo), f(hi))
    for point in ({0.0} if power == 0 else {-np.sqrt(power), np.sqrt(power)}):
        result = np.maximum(result, np.where((lo <= point) & (point <= hi), f(point), 0.0))
    return result


def hermite_phi_max(lo, hi, order):
    def f(x):
        h = (1.0, x, x * x - 1.0, x * x * x - 3.0 * x)[order]
        return np.abs(h) * phi(x)
    roots = ([0.0], [-1.0, 1.0], [-np.sqrt(3.0), 0.0, np.sqrt(3.0)],
             [-np.sqrt(3.0 + np.sqrt(6.0)), -np.sqrt(3.0 - np.sqrt(6.0)),
              np.sqrt(3.0 - np.sqrt(6.0)), np.sqrt(3.0 + np.sqrt(6.0))])[order]
    result = np.maximum(f(lo), f(hi))
    for point in roots:
        result = np.maximum(result, np.where((lo <= point) & (point <= hi), f(point), 0.0))
    return result


def derivative_bounds(al, ah, bl, bh, tl, th):
    tan_lo, tan_hi = np.tan(tl), np.tan(th)
    tangent = max(abs(tan_lo), abs(tan_hi))
    sec_hi = max(1.0 / np.cos(tl), 1.0 / np.cos(th))
    sec_lo = 1.0 if tl <= 0 <= th else min(1.0 / np.cos(tl), 1.0 / np.cos(th))

    def conditional_range(xl, xh, yl, yh):
        # y sec(theta) - x tan(theta), using conservative interval products.
        l1, h1 = product_interval(yl, yh, sec_lo, sec_hi)
        l2, h2 = product_interval(xl, xh, tan_lo, tan_hi)
        return l1 - h2, h1 - l2

    def mean_bound(xl, xh, yl, yh):
        dl, dh = conditional_range(xl, xh, yl, yh)
        m = [hermite_phi_max(xl, xh, n) for n in range(4)]
        d = [hermite_phi_max(dl, dh, n) for n in range(3)]
        return (m[3] * ndtr(dh) + 3 * tangent * m[2] * d[0]
                + 3 * tangent ** 2 * m[1] * d[1] + tangent ** 3 * m[0] * d[2])

    def theta_bound(xl, xh, yl, yh):
        # y here is the conditional standardized x; b is the second argument.
        cl, ch = conditional_range(yl, yh, xl, xh)
        y = [power_phi_max(cl, ch, n) for n in range(7)]
        b = [power_phi_max(yl, yh, n) for n in range(4)]
        t = tangent
        return (t ** 3 * b[0] * y[6] + 3 * t ** 2 * b[1] * y[5]
                + 3 * t * ((1 + 3 * t ** 2) * b[0] + b[2]) * y[4]
                + (b[3] + (3 + 18 * t ** 2) * b[1]) * y[3]
                + 4 * t * (3 * b[2] + (2 + 3 * t ** 2) * b[0]) * y[2]
                + ((5 + 12 * t ** 2) * b[1] + 3 * b[3]) * y[1]
                + 3 * t * b[2] * y[0])

    return (mean_bound(al, ah, bl, bh), mean_bound(bl, bh, al, ah),
            np.minimum(theta_bound(al, ah, bl, bh), theta_bound(bl, bh, al, ah)))


def generate(output):
    begin = time.perf_counter()
    minimum, step, count = -5.0, 0.1, 101
    theta_minimum, theta_step, theta_count = -1.28, 0.02, 129
    coordinates = minimum + step * np.arange(count)
    theta = theta_minimum + theta_step * np.arange(theta_count)
    rows, columns = np.tril_indices(count)
    a, b = coordinates[columns], coordinates[rows]
    independent = ndtr(a) * ndtr(b)
    full = np.empty((theta_count, count, count), dtype=np.float64)
    node_errors = np.empty_like(full)
    rules = [leggauss(order) for order in (64, 96)]
    for k, angle in enumerate(theta):
        estimates = []
        for nodes, weights in rules:
            angles = angle * (nodes + 1.0) / 2.0
            sine = np.sin(angles)
            cosine = np.cos(angles)
            exponent = -0.5 * (((a[:, None] - sine * b[:, None]) / cosine) ** 2
                               + b[:, None] ** 2)
            integral = np.exp(exponent) @ weights * angle / (4.0 * np.pi)
            estimates.append(independent + integral)
        p = np.clip(estimates[1], 0.0, 1.0)
        stored = p.astype(np.float32)
        error = np.maximum(1e-13, 16 * np.abs(estimates[0] - estimates[1])) + np.abs(p - stored)
        full[k, rows, columns] = full[k, columns, rows] = stored
        node_errors[k, rows, columns] = node_errors[k, columns, rows] = error
    print(f"Generated {theta_count * len(a):,} nodes in {time.perf_counter()-begin:.2f}s", flush=True)

    nc, ntc = count - 3, theta_count - 3
    max_node_error = np.zeros((ntc, nc, nc))
    for kt, ib, ia in itertools.product(range(4), repeat=3):
        np.maximum(max_node_error, node_errors[kt:kt+ntc, ib:ib+nc, ia:ia+nc], out=max_node_error)
    al, ah = coordinates[:-3][None, :], coordinates[3:][None, :]
    bl, bh = coordinates[:-3][:, None], coordinates[3:][:, None]
    cell_errors = np.empty_like(max_node_error)
    coefficient = 3.0 / 128.0
    for k in range(ntc):
        ma, mb, mt = derivative_bounds(al, ah, bl, bh, theta[k], theta[k+3])
        axis_errors = (coefficient * step ** 4 * ma, coefficient * step ** 4 * mb,
                       coefficient * theta_step ** 4 * mt)
        bound = np.full((nc, nc), np.inf)
        for i, j, l in itertools.permutations(range(3)):
            bound = np.minimum(bound, axis_errors[i] + 1.25 * axis_errors[j] + 1.25 ** 2 * axis_errors[l])
        cell_errors[k] = bound * (1 + 1e-12) + 1.25 ** 3 * max_node_error[k] + 1e-18
    cell_errors = np.nextafter(cell_errors.astype(np.float32), np.float32(np.inf))
    cell_rows, cell_columns = np.tril_indices(nc)
    values = full[:, rows, columns].astype('<f4').ravel()
    errors = cell_errors[:, cell_rows, cell_columns].astype('<f4').ravel()
    header = struct.pack('<8s4I4d2Q', b'GPBCDF1\0', 1, count, theta_count, 0,
                         minimum, step, theta_minimum, theta_step, len(values), len(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        stream.write(header)
        stream.write(values.tobytes())
        stream.write(errors.tobytes())
    print(f"Wrote {output} ({output.stat().st_size:,} bytes) in {time.perf_counter()-begin:.2f}s", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/gpis/bivariate-normal-cdf-v1.bin')
    generate(parser.parse_args().output)
