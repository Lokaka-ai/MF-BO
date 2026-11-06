#!/usr/bin/env python3
"""Download pinned official constructions and prepare public mathematical tasks."""
import argparse
import contextlib
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import requests
from tasks.alphaevolve.registry import TASKS, SOURCE_COMMIT, SOURCE_URL, NOTEBOOK_SHA256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.tolist())+"\n"
    if not path.exists() or path.read_text() != text:
        temp = path.with_suffix(path.suffix+".tmp")
        temp.write_text(text)
        temp.replace(path)


def initial(task, reference):
    rng = np.random.default_rng(42)
    key, n, tid = task["output_key"], task["n"], task["id"]
    if tid == "hexagon_packing":
        return {"hexagons": [[(i%4-1.5)*2.2, (i//4-1)*2.2, 0] for i in range(11)], "outer": [0, 0, 10, 0]}
    if key == "circles":
        cols, rows = (6, 5) if n == 26 else (7, 3)
        width, height = (1, 1) if n == 26 else (1.4, .6)
        r = .9*min(width/cols, height/rows)/2
        return {key: [[(i%cols+.5)*width/cols, (i//cols+.5)*height/rows, r] for i in range(n)]}
    if tid == "kissing_number_d11":
        return {key: np.concatenate((np.eye(11), -np.eye(11))).tolist()}
    if tid == "heilbronn_triangle":
        uv = rng.uniform(size=(n, 2))
        uv[uv.sum(axis=1)>1] = 1-uv[uv.sum(axis=1)>1]
        return {key: np.column_stack((uv[:, 0]+.5*uv[:, 1], np.sqrt(3)/2*uv[:, 1])).tolist()}
    if tid == "heilbronn_convex":
        return {key: rng.uniform(size=(n, 2)).tolist()}
    if tid == "max_min_ratio":
        return {key: [[i%4, i//4] for i in range(16)]}
    if key == "integers":
        return {key: list(range(64 if tid.endswith("1") else 256))}
    if key == "coefficients":
        # Published pre-AlphaEvolve construction, not AlphaEvolve's result.
        return {key: [-113/100, 1/25, 1/3240]}
    return {key: [.5 if tid == "minimum_overlap" else 1.] * len(reference[key])}


def contract(task):
    tid, key, n = task["id"], task["output_key"], task["n"]
    specific = {
        "hexagon_packing": "Return {'hexagons': eleven [x,y,angle_degrees] rows, 'outer': [center_x,center_y,side_length,angle_degrees]}. Every inner hexagon has side 1. No overlaps; all vertices contained. Minimize outer side length.",
        "circle_packing_square": "Return {'circles': 26 [x,y,radius] rows}, radii > 0, disjoint circles fully within [0,1]^2. Maximize sum of radii.",
        "circle_packing_rectangle": "Return {'circles': 21 [x,y,radius] rows}, radii > 0, disjoint circles. Their smallest axis-aligned bounding rectangle must have width+height <= 2. Maximize sum of radii.",
        "heilbronn_triangle": "Return {'points': 11 [x,y] rows} in the triangle (0,0),(1,0),(0.5,sqrt(3)/2). Maximize the minimum area of ALL point triples divided by sqrt(3)/4.",
        "heilbronn_convex": "Return {'points': 13 [x,y] rows}. Maximize min triangle area over ALL point triples divided by convex hull area; hull must have positive area.",
        "kissing_number_d11": "Return {'points': N nonzero 11-dimensional vectors}, 2 <= N <= 5000. Each is normalized to length 2; ALL pairwise normalized distances must be >= 2. Maximize N.",
        "max_min_ratio": "Return {'points': 16 distinct [x,y] rows}. Minimize (max pairwise distance / min pairwise distance)^2 (the SQUARED ratio).",
        "uncertainty_inequality": "Return {'coefficients': three real numbers c0,c1,c2}. Form P=sum(c_i H_{4i}) plus a derived H12 term imposing P(0)=0, and choose positive leading sign. Minimize largest positive sign-changing root(P/x^2)^2/(2*pi). H are physicists' Hermite polynomials. The host uses exact rational root isolation, not sampled signs.",
        "minimum_overlap": "Return {'heights': a finite sequence, each in [0,1], with mean exactly 1/2}. Minimize 2/N*max(correlate(h,1-h,'full')). Mean tolerance 1e-10. Length 2..10000.",
        "first_autocorrelation": "Return {'heights': N nonnegative heights, positive sum}. Minimize 2*N*max(convolve(h,h))/sum(h)^2; N can vary from 2 to 10000.",
        "third_autocorrelation": "Return {'heights': N signed real heights, nonzero sum}. Minimize 2*N*abs(max(convolve(h,h)))/sum(h)^2. Absolute value is OUTSIDE max. Length 2..10000.",
        "third_autocorrelation_variant": "Return {'heights': N signed real heights, nonzero sum}. Minimize 2*N*max(abs(convolve(h,h)))/sum(h)^2. Absolute value is INSIDE max. Length 2..10000.",
        "second_autocorrelation": "Return {'heights': N nonnegative heights, positive sum}. Let c=convolve(h,h), y=[0]+list(c)+[0]. Maximize sum(y[:-1]^2+y[:-1]*y[1:]+y[1:]^2)/(3*sum(abs(c))*max(abs(c))). This exactly integrates the piecewise-linear autoconvolution squared. Length 2..10000.",
    }
    if tid.startswith("sums_differences"):
        return ("Return {'integers': U}, a set of distinct nonnegative integers including 0. Maximize "
                "1+log(|U-U|/|U+U|)/log(2*max(U)+1). Cardinality may vary. "
                "Runtime limits: 2..100000 entries, max(U)<=1e9. Prefer compact sets for fast grading. "
                "Rows 1 and 2 use the same official inequality with different reference constructions and initial sets; no cardinality bonus is added.")
    return specific[tid]


def prepare(data_dir):
    source = ROOT / ".cache/alphaevolve/source"
    source.mkdir(parents=True, exist_ok=True)
    with (source/"prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ("mathematical_results.ipynb", "LICENSE", "README.md"):
            path = source/name
            if not path.exists():
                response = requests.get(f"{SOURCE_URL}/{name}", timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
        notebook = source/"mathematical_results.ipynb"
        if digest(notebook) != NOTEBOOK_SHA256:
            raise ValueError("Official notebook SHA256 mismatch; refusing changed source")
        cells = json.loads(notebook.read_text())["cells"]
        manifest = {"version": 1, "source_commit": SOURCE_COMMIT, "notebook_sha256": NOTEBOOK_SHA256, "tasks": {}}
        for tid, task in TASKS.items():
            ns = {"np": np}
            # Only fixed, hash-verified upstream DATA cells, never candidate code.
            with contextlib.redirect_stdout(io.StringIO()):
                exec("".join(cells[task["data_cell"]]["source"]), ns)
            if tid == "hexagon_packing":
                reference = {"hexagons": ns["inner_hex_data"], "outer": [*ns["outer_hex_center"], ns["outer_hex_side_length"], ns["outer_hex_angle_degrees"]]}
            else:
                variable = {"circle_packing_square": "construction_1", "circle_packing_rectangle": "circles",
                            "heilbronn_convex": "construction_1", "heilbronn_triangle": "found_points",
                            "kissing_number_d11": "sphere_centers", "sums_differences_1": "solution_1", "sums_differences_2": "solution_2",
                            "uncertainty_inequality": "coefficients_alphaevolve", "first_autocorrelation": "step_function_heights_1",
                            "third_autocorrelation_variant": "height_sequence_4", "third_autocorrelation": "height_sequence_3",
                            "second_autocorrelation": "heights_sequence_2", "max_min_ratio": "construction_1", "minimum_overlap": "best_sequence"}[tid]
                reference = {task["output_key"]: ns[variable]}
            public, private = data_dir/tid/"prepared/public", data_dir/tid/"prepared/private"
            # The agent sees problem definitions and a simple feasible start.
            # Official AlphaEvolve outputs remain grader fixtures, not search inputs.
            problem = {k: v for k, v in task.items() if k not in ("reference_score", "data_cell")}
            problem.update(contract=contract(task), source_commit=SOURCE_COMMIT)
            save(public/"problem.json", problem)
            save(public/"initial.json", initial(task, reference))
            description = (f"# {task['title']}\n\n"+contract(task)+"\n\n## Official problem definition\n\n"
                           +"".join(cells[task["description_cell"]]["source"])+"\n")
            (public/"description.md").write_text(description)
            save(private/"reference.json", reference)
            save(private/"expected.json", {"metric": task["reference_score"], "abs_tolerance": 2e-6})
            manifest["tasks"][tid] = {str(p.relative_to(data_dir)): digest(p) for p in [public/"problem.json", public/"initial.json", public/"description.md", private/"reference.json", private/"expected.json"]}
        save(data_dir/"manifest.json", manifest)
    print(f"Prepared {len(TASKS)} tasks at {data_dir}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/alphaevolve")
    prepare(parser.parse_args().data_dir.resolve())
