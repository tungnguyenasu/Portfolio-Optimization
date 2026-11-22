"""One helper used by every script to print comparisons in the same format.

    max|diff| = max_i |mine_i - ref_i|
    rel       = ||mine - ref||_2 / ||ref||_2
"""
import numpy as np


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||.

    Returns (abs_err, rel_err) so scripts can reuse the numbers in tables.
    """
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
    return abs_err, rel_err
