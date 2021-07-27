import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from typing import List, Tuple
from pathlib import Path
import argparse
import json

import globalflow as gflow


DATA_DIR = Path(__file__).parent / ".." / "etc" / "data"


def draw_skeleton(ax, xys: np.ndarray, limbs: List[Tuple[int, int]], color="k"):
    for limb in limbs:
        plt.plot(xys[limb[0], 0], xys[limb[1], 1], color=color)
    plt.scatter(xys[:, 0], xys[:, 1], color=color)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser("Determine trajectories from global flow.")
    parser.add_argument(
        "-skeleton",
        type=Path,
        help="Skeleton definition",
        default=DATA_DIR / "skeleton.json",
    )
    parser.add_argument("keypoints", type=Path, help="Keypoints")
    parser.add_argument("--show", action="store_true", help="Show result")

    args = parser.parse_args()
    assert args.skeleton.is_file()
    assert args.keypoints.is_file()

    skel = json.load(open(args.skeleton, "r"))
    kpts = json.load(open(args.keypoints, "r"))

    for fname, objs in kpts.items():
        fig, ax = plt.subplots()
        cycle = ax._get_lines.prop_cycler
        for obj in objs:
            xys = np.array(obj["keypoints"]).reshape(-1, 3)
            draw_skeleton(ax, xys, skel["limbs"], color=next(cycle)["color"])
        plt.show()


if __name__ == "__main__":
    main()
