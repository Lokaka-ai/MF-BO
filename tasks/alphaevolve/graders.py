"""Host-owned mathematical verification. No candidate-supplied scores.

Formulas and Hermite verifier adapted from Google DeepMind's Apache-2.0
mathematical_results.ipynb; see SOURCE.md and vendor/LICENSE.
"""
import itertools
import math

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from tasks.alphaevolve.registry import task_spec

TOL = 1e-9


def array(value, *, ndim, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("Construction must contain real numbers, not booleans/strings")
    a = raw.astype(np.float64)
    if a.ndim != ndim or a.size == 0 or a.size > 2_000_000:
        raise ValueError("Empty, oversized or incorrectly shaped construction")
    if shape is not None and a.shape != shape:
        raise ValueError(f"Expected shape {shape}, received {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("Construction contains NaN or infinity")
    return a


def triangle_areas(points):
    triples = np.asarray(list(itertools.combinations(range(len(points)), 3)))
    a, b, c = (points[triples[:, i]] for i in range(3))
    return np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])) / 2


def sum_difference_counts(u):
    """Exact integer counts; FFT only where rounding is numerically unambiguous.

    Large sparse official construction 2 needs boolean support arrays (3.3 GB),
    not a prohibitively large floating-point FFT or an O(n²) Python set.
    """
    maximum = int(u[-1])
    if maximum <= 4_000_000:
        indicator = np.zeros(maximum + 1)
        indicator[u] = 1
        counts = []
        for other in (indicator, indicator[::-1]):
            conv = fftconvolve(indicator, other)
            rounded = np.rint(conv)
            if np.max(np.abs(conv - rounded)) > 1e-5:
                raise ArithmeticError("Ambiguous FFT integer counts")
            counts.append(int(np.count_nonzero(rounded > 0)))
        return counts[0], counts[1]
    sums = np.zeros(2 * maximum + 1, dtype=bool)
    diffs = np.zeros(2 * maximum + 1, dtype=bool)
    for value in u:
        sums[value + u] = True
        diffs[value - u + maximum] = True
    return int(np.count_nonzero(sums)), int(np.count_nonzero(diffs))


def hermite_bound(coeffs):
    """Official rational-polynomial/sign-change construction, not a grid proxy."""
    import sympy as s
    x = s.symbols("x")
    hs = [s.polys.orthopolys.hermite_poly(4*i, x) for i in range(len(coeffs)+1)]
    partial = sum(s.Rational(float(c))*h for c, h in zip(coeffs, hs))
    polynomial = s.expand(partial - partial.subs(x, 0)/hs[-1].subs(x, 0)*hs[-1])
    if polynomial == 0:
        raise ValueError("Zero Hermite polynomial")
    if s.LC(s.Poly(polynomial, x)) < 0:
        polynomial = -polynomial
    if polynomial.subs(x, 0) != 0:
        raise ValueError("Hermite constraint P(0)=0 failed")
    q = s.exquo(polynomial, x**2)
    largest = s.Rational(0)
    for root in s.real_roots(q, x):
        r = root.eval_rational(n=200)
        epsilon = s.Rational("1e-198")
        if q.subs(x, r+epsilon)*q.subs(x, r-epsilon) < 0:
            largest = max(largest, r)
    if largest <= 0:
        raise ValueError("No positive sign-changing root")
    return float(largest**2) / (2*np.pi)


def hex_vertices(x, y, side, angle):
    theta = np.radians(angle) + np.arange(6)*np.pi/3
    return np.column_stack((x+side*np.cos(theta), y+side*np.sin(theta)))


def hexagon_score(result):
    inside = array(result["hexagons"], ndim=2, shape=(11, 3))
    outer = array(result["outer"], ndim=1, shape=(4,))
    if outer[2] <= 0:
        raise ValueError("Outer side must be positive")
    boundary = hex_vertices(*outer)
    polygons = [hex_vertices(x, y, 1, angle) for x, y, angle in inside]
    for poly in polygons:
        for i in range(6):
            edge = boundary[(i+1) % 6]-boundary[i]
            offsets = poly-boundary[i]
            signed = (edge[0]*offsets[:, 1]-edge[1]*offsets[:, 0])/np.linalg.norm(edge)
            if signed.min() < -TOL:
                raise ValueError("Unit hexagon lies outside outer hexagon")
    for p, q in itertools.combinations(polygons, 2):
        separated = False
        for polygon in (p, q):
            edges = np.roll(polygon, -1, axis=0)-polygon
            for edge in edges:
                axis = np.array([-edge[1], edge[0]])/np.linalg.norm(edge)
                pp, qq = p@axis, q@axis
                if min(pp.max(), qq.max()) - max(pp.min(), qq.min()) <= TOL:
                    separated = True
                    break
            if separated:
                break
        if not separated:
            raise ValueError("Unit hexagons overlap")
    return float(outer[2])


def score(task_id, result):
    spec = task_spec(task_id)
    if not isinstance(result, dict) or spec["output_key"] not in result:
        raise ValueError(f"Return a dict containing {spec['output_key']!r}")
    key = spec["output_key"]
    if task_id == "hexagon_packing":
        value = hexagon_score(result)
    elif key == "circles":
        a = array(result[key], ndim=2, shape=(spec["n"], 3))
        if (a[:, 2] <= 0).any():
            raise ValueError("All radii must be positive")
        i, j = np.triu_indices(len(a), 1)
        if (pdist(a[:, :2]) - a[i, 2] - a[j, 2] < -TOL).any():
            raise ValueError("Circles overlap")
        lower = (a[:, :2]-a[:, 2, None]).min(axis=0)
        upper = (a[:, :2]+a[:, 2, None]).max(axis=0)
        if task_id.endswith("square"):
            if lower.min() < -TOL or upper.max() > 1+TOL:
                raise ValueError("Circles are outside the unit square")
        elif (upper-lower).sum() > 2+TOL:
            raise ValueError("Bounding rectangle perimeter exceeds 4")
        value = float(a[:, 2].sum())
    elif task_id.startswith("heilbronn"):
        p = array(result[key], ndim=2, shape=(spec["n"], 2))
        if task_id.endswith("triangle"):
            x, y = p.T
            if min(y.min(), (np.sqrt(3)*x-y).min(), (np.sqrt(3)*(1-x)-y).min()) < -TOL:
                raise ValueError("Points outside the equilateral triangle")
            area = np.sqrt(3)/4
        else:
            area = ConvexHull(p).volume
        value = float(triangle_areas(p).min()/area)
    elif task_id == "max_min_ratio":
        p = array(result[key], ndim=2, shape=(16, 2))
        distances = pdist(p)
        if distances.min() <= 0:
            raise ValueError("Duplicate points")
        value = float((distances.max()/distances.min())**2)
    elif task_id == "kissing_number_d11":
        p = array(result[key], ndim=2)
        if p.shape[1] != 11 or not 2 <= len(p) <= 5000:
            raise ValueError("Expected 2..5000 points of dimension 11")
        norms = np.linalg.norm(p, axis=1)
        if (norms <= 0).any():
            raise ValueError("Zero sphere center")
        # Directly verify the resulting unit-sphere configuration. This also
        # admits real configurations beyond the sufficient integer lemma.
        normalized = 2*(p/norms[:, None])
        if pdist(normalized).min() < 2-TOL:
            raise ValueError("Normalized kissing spheres overlap")
        value = len(p)
    elif task_id.startswith("sums_differences"):
        a = array(result[key], ndim=1)
        if len(a) > 100_000 or (a != np.floor(a)).any() or a.min() != 0 or a.max() > 1_000_000_000:
            raise ValueError("U must contain 0 and at most 100000 nonnegative integers <= 1e9")
        u = np.unique(a.astype(np.int64))
        if len(u) < 2 or len(u) != len(a):
            raise ValueError("Expected at least two distinct integers")
        plus, minus = sum_difference_counts(u)
        value = 1+math.log(minus/plus)/math.log(2*int(u[-1])+1)
    elif task_id == "uncertainty_inequality":
        value = hermite_bound(array(result[key], ndim=1, shape=(3,)))
    else:
        h = array(result[key], ndim=1)
        if not 2 <= len(h) <= 10000:
            raise ValueError("Expected 2..10000 step heights")
        if task_id == "minimum_overlap":
            if h.min() < 0 or h.max() > 1 or abs(h.mean()-.5) > 1e-10:
                raise ValueError("Overlap heights must be in [0,1] and have mean 1/2")
            value = float(np.correlate(h, 1-h, "full").max()*2/len(h))
        else:
            if task_id in ("first_autocorrelation", "second_autocorrelation") and h.min() < 0:
                raise ValueError("This inequality requires nonnegative heights")
            scale = np.max(np.abs(h))
            if scale == 0:
                raise ValueError("Zero function")
            h = h/scale
            conv = np.convolve(h, h)
            if abs(h.sum()) < 1e-12:
                raise ValueError("Function has zero integral")
            if task_id == "second_autocorrelation":
                y = np.r_[0., conv, 0.]
                integral2 = np.sum(y[:-1]**2+y[:-1]*y[1:]+y[1:]**2)/(3*(len(conv)+1))
                value = float(integral2/((np.abs(conv).sum()/(len(conv)+1))*np.abs(conv).max()))
            else:
                peak = np.abs(conv).max() if task_id.endswith("variant") else abs(conv.max())
                value = float(2*len(h)*peak/h.sum()**2)
    if not math.isfinite(value):
        raise ValueError("Nonfinite mathematical score")
    return {"valid": True, "metric": value, "maximize": spec["maximize"], "task_id": task_id}
