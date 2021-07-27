import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import globalflow as gflow
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

DATA_DIR = Path(__file__).parent / ".." / "etc" / "data"


def draw_skeleton(ax, xys: np.ndarray, limbs: List[Tuple[int, int]], color="k"):
    """Draws the skeleton of a particular instance"""
    for limb in limbs:
        plt.plot(
            [xys[limb[0], 0], xys[limb[1], 0]],
            [xys[limb[0], 1], xys[limb[1], 1]],
            color="w",
        )
    plt.scatter(xys[:, 0], xys[:, 1], color=color, s=5)


def draw_instances(
    ax,
    objs: List[Dict],
    ids: List[int],
    limbs: List[Tuple[int, int]],
    imagepath: Path = None,
    cmap=None,
):
    """Draws instance for a particular frame"""
    if cmap is None:
        cmap = plt.get_cmap("tab10", 10)
    if imagepath is not None:
        img = plt.imread(imagepath)
        ax.imshow(img)
        for oidx, obj in enumerate(objs):
            xys = np.array(obj["keypoints"]).reshape(-1, 3)
            color = cmap(ids[oidx]) if ids[oidx] != -1 else "w"
            draw_skeleton(ax, xys, limbs, color=color)


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
    parser.add_argument("--show", action="store_true", help="Show result")
    parser.add_argument("-keypoints", type=Path, help="Keypoints")
    parser.add_argument(
        "-imagedir", type=Path, help="Optional image dir.", required=False
    )

    args = parser.parse_args()
    assert args.skeleton.is_file()
    assert args.keypoints.is_file()
    assert args.imagedir is None or args.imagedir.is_dir()

    skel = json.load(open(args.skeleton, "r"))
    kpts = json.load(open(args.keypoints, "r"))

    for fname, objs in kpts.items():
        fig, ax = plt.subplots()
        imgpath = None
        if args.imagedir is not None:
            imgpath = args.imagedir / fname
        draw_instances(ax, objs, np.arange(len(objs)), skel["limbs"], imgpath)
        plt.show()


if __name__ == "__main__":
    main()
