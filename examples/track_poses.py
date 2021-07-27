import argparse
import json
import logging
from math import fabs
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import globalflow as gflow
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import warnings

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


@dataclass(frozen=True)
class Detection:
    center: np.ndarray
    minc: np.ndarray
    maxc: np.ndarray
    area: float


def quiet_divide(a, b):
    """Quiet divide function that does not warn about (0 / 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.true_divide(a, b)


def boxiou(det1, det2):
    """Computes IOU of two rectangles."""
    a_min, a_max = det1.minc, det1.maxc
    b_min, b_max = det2.minc, det2.maxc
    # Compute intersection.
    i_min = np.maximum(a_min, b_min)
    i_max = np.minimum(a_max, b_max)
    i_size = np.maximum(i_max - i_min, 0)
    i_vol = np.prod(i_size, axis=-1)
    # Get volume of union.
    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = np.prod(a_size, axis=-1)
    b_vol = np.prod(b_size, axis=-1)
    u_vol = a_vol + b_vol - i_vol
    return np.where(
        i_vol == 0, np.zeros_like(i_vol, dtype=float), quiet_divide(i_vol, u_vol)
    )


def find_trajectories(args, kpts: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    timeseries = []
    fnames = []
    for t, (fname, objs) in enumerate(kpts.items()):
        fnames.append(fname)
        tdata = []
        for obj in objs:
            xys = np.array(obj["keypoints"]).reshape(-1, 3)
            minc = np.min(xys[:, :2], axis=0)
            maxc = np.max(xys[:, :2], axis=0)
            c = (minc + maxc) * 0.5
            area = (maxc[0] - minc[0]) * (maxc[1] - minc[1])
            tdata.append(Detection(c, minc, maxc, area))
        timeseries.append(tdata)
        # if t > 200:
        #     break

    def logp_trans(xi: gflow.FlowNode, xj: gflow.FlowNode):
        """Log-probability of pairing xi(t-1) with xj(t)."""
        iou = boxiou(xi.obs, xj.obs)
        # tdiff = xj.time_index - xi.time_index
        # print(iou, xi.obs.area, xj.obs.area)
        return np.log(iou + 1e-5)
        # tdiff = xj.time_index - xi.time_index
        # prob = (
        #     # scipy.stats.norm.logpdf(
        #     #     xj.obs.center[0], loc=xi.obs.center[0], scale=tdiff * 5
        #     # )
        #     # # + np.log(0.1)
        #     # + scipy.stats.norm.logpdf(
        #     #     xj.obs.center[1], loc=xi.obs.center[1], scale=tdiff * 5
        #     # )
        #     +scipy.stats.norm.logpdf(xj.obs.area, loc=xi.obs.area, scale=100)
        #     # + np.log(0.1)
        # )
        # print(xi.obs.center, xj.obs.center, xj.obs.area, xi.obs.area, prob)
        # return prob

    def logp_enter(xi: gflow.FlowNode):
        """Log-probability of xi(t) appearing."""
        return 0.0 if xi.time_index == 0 else np.log(1e-2)

    def logp_exit(xi: gflow.FlowNode):
        """Log-probability of xi(t) disappearing."""
        return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(1e-2)

    flow = gflow.GlobalFlowMOT(
        timeseries,
        logp_enter,
        logp_exit,
        logp_trans,
        gflow.default_logp_fp_fn(beta=0.01),
        num_skip_layers=0,
    )

    # Solve the problem
    flowdict, ll = flow.solve((1, 10))
    trajs = gflow.find_trajectories(flow, flowdict)

    print(
        "optimum: log-likelihood",
        ll,
        "number of trajectories",
        len(trajs),
    )  # optimum: log-likelihood 16.76 number of trajectories 2

    trajectories = gflow.find_trajectories(flow, flowdict)
    indices = gflow.label_observations(timeseries, trajectories)
    return {fname: ids for fname, ids in zip(fnames, indices)}, len(trajs)


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

    trajdict, num_trajectories = find_trajectories(args, kpts)

    plt.ion()
    fig, ax = plt.subplots()
    for fname, objs in kpts.items():
        ax.cla()
        imgpath = None
        if args.imagedir is not None:
            imgpath = args.imagedir / fname
        draw_instances(ax, objs, trajdict[fname], skel["limbs"], imgpath)
        plt.draw()
        plt.pause(0.003)
    plt.ioff()


if __name__ == "__main__":
    main()
