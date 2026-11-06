"""The fifteen rows of MLEvolve Table 2, mapped to DeepMind's notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "4226acbf237ff9ad10ba7673a2af127a2d8a5971"
NOTEBOOK_SHA256 = "2cce2543e48c89aa3e91614272a698a0147dd2548ea11cf92f1292b7435d38ff"
SOURCE_URL = f"https://raw.githubusercontent.com/google-deepmind/alphaevolve_results/{SOURCE_COMMIT}"

# The two sums/differences rows refer to the two published constructions of
# the SAME inequality. Neither the paper nor the official verifier fixes |U|.
# We therefore do not invent different objectives or cardinality constraints.
ROWS = [
    ("hexagon_packing", "Packing 11 unit hexagons in a hexagon", False, 85, 88, "hexagons", 11, 3.930092),
    ("circle_packing_square", "Packing 26 circles in a unit square", True, 115, 118, "circles", 26, 2.6358627564),
    ("circle_packing_rectangle", "Packing 21 circles in a perimeter-4 rectangle", True, 124, 125, "circles", 21, 2.3658321334),
    ("heilbronn_convex", "Heilbronn convex region, 13 points", True, 103, 106, "points", 13, .0309368890),
    ("heilbronn_triangle", "Heilbronn triangle, 11 points", True, 99, 100, "points", 11, .03652988988003016),
    ("kissing_number_d11", "Kissing number in dimension 11", True, 112, 113, "points", None, 593),
    ("sums_differences_1", "Sums and differences, reference construction 1", True, 80, 81, "integers", None, 1.1479889651),
    ("sums_differences_2", "Sums and differences, reference construction 2", True, 80, 81, "integers", None, 1.1584172816),
    ("uncertainty_inequality", "Uncertainty inequality (Hermite formulation)", False, 67, 68, "coefficients", 3, .3520991044225),
    ("first_autocorrelation", "First autocorrelation inequality", False, 52, 53, "heights", None, 1.5052939684),
    ("third_autocorrelation_variant", "Third autocorrelation inequality, variant", False, 64, 65, "heights", None, 1.4687620697),
    ("third_autocorrelation", "Third autocorrelation inequality", False, 60, 61, "heights", None, 1.4556427954),
    ("second_autocorrelation", "Second autocorrelation inequality", True, 56, 57, "heights", None, .8962799442),
    ("max_min_ratio", "Squared max/min distance ratio, 16 points in 2D", False, 92, 93, "points", 16, 12.88926611203),
    ("minimum_overlap", "Erdos minimum overlap problem", False, 76, 78, "heights", None, .3809230351),
]
TASKS = {
    row[0]: dict(zip(("id", "title", "maximize", "description_cell", "data_cell", "output_key", "n", "reference_score"), row))
    for row in ROWS
}


def task_spec(task_id):
    if task_id not in TASKS:
        raise ValueError(f"Unknown AlphaEvolve task: {task_id}")
    return TASKS[task_id].copy()
