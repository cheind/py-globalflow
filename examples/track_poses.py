"""Example that tracks 2D human poses over time using py-globalflow.

This application takes as input a JSON dictionary that contains a 
mapping from filenames to a list of detected objects. Each object is expected 
to have a  `keypoints` field that is a 1D list of [x0,y0,s0,x1,y1,s1,...] 
coordinates. The filenames are expected to be ordered in time.

From this information (and a couple of optional parameters) the program
outputs a trajectory dictionary that maps from filenames to a list of 
instance ids (same length as the number of objects in that frame). Valid
instance ids are >=0, -1 is reserved for a not used detection.

The link-probabilities are defined per default on geometric properties
of the detected 2D keypoints and hence tracking does not incorporate any
color information. Appearance information can be optionally included 
in terms of ReID features. See `reid_features.py` for generating deeply learned
ReID descriptors from detections.

For rendering the application expects a description of limbs. By default
COCO is assumed. A valid description can be found in etc/data.

The found trajectories are saved as follows. Let L be the computed trajectories.
Then, L[t] refers observation at time t. L[t][j] is the trajectory id for the 
j-th observation at time t. This trajectory id might be -1 to signal a 
non-valid observation.
"""
import argparse
import json
import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import globalflow as gflow
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

DATA_DIR = Path(__file__).parent / ".." / "etc" / "data"
TMP_DIR = Path(__file__).parent / ".." / "tmp"


def draw_skeleton(ax, xys: np.ndarray, limbs: List[Tuple[int, int]], color="k"):
    """Draws the skeleton of a particular instance"""
    x = xys[:, 0]
    y = xys[:, 1]
    for limb in limbs:
        ax.plot(x[limb], y[limb], color=color, linewidth=1, alpha=0.5, zorder=1)
    ax.scatter(x, y, color=color, s=8, zorder=2)


def draw_instances(
    ax,
    objs: List[Dict],
    ids: List[int],
    limbs: List[Tuple[int, int]],
    image: np.ndarray = None,
    max_instances: int = 10,
    cmap=None,
):
    """Draws instance for a particular frame"""
    if cmap is None:
        cmap = plt.get_cmap("jet", max_instances)
    if image is not None:
        ax.imshow(image)
    objs = [obj for iid, obj in zip(ids, objs) if iid > -1]
    ids = [iid for iid in ids if iid > -1]
    for oidx, obj in enumerate(objs):
        xys = np.array(obj["keypoints"]).reshape(-1, 3)
        draw_skeleton(ax, xys, limbs, color=cmap(ids[oidx]))


@dataclass(frozen=True)
class Detection:
    center: np.ndarray
    minc: np.ndarray
    maxc: np.ndarray
    area: float
    reid: Optional[np.ndarray] = None


@dataclass
class Stats:
    minc: np.ndarray
    maxc: np.ndarray
    num_max_det: int


def quiet_divide(a, b):
    """Quiet divide function that does not warn about (0 / 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.true_divide(a, b)


def boxiou(det1, det2):
    """Computes IOU of two rectangles. Taken from
    https://github.com/cheind/py-motmetrics/blob/6597e8a4ed398b9f14880fa76de26bc43d230836/motmetrics/distances.py#L64
    """
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


def find_trajectories(
    args, kpts: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Stats]:
    timeseries = []
    fnames = []

    reid_feature_map = None
    if args.reidpath is not None:
        with open(args.reidpath, "rb") as f:
            data = pickle.load(f)
        reid_feature_map = data["compressed_features"]
        print(reid_feature_map)

    stats = Stats(minc=np.array([1e3] * 2), maxc=np.array([-1e3] * 2), num_max_det=0)
    for t, (fname, objs) in enumerate(kpts.items()):
        fnames.append(fname)
        tdata = []
        if reid_feature_map is not None:
            reid_features = reid_feature_map[fname]
            print(fname, reid_features)
        else:
            reid_features = [None] * len(objs)
        for oidx, obj in enumerate(objs):
            xys = np.array(obj["keypoints"]).reshape(-1, 3)
            minc = np.min(xys[:, :2], axis=0)
            maxc = np.max(xys[:, :2], axis=0)
            c = (minc + maxc) * 0.5
            area = (maxc[0] - minc[0]) * (maxc[1] - minc[1])

            tdata.append(Detection(c, minc, maxc, area, reid_features[oidx]))
            stats.minc = np.minimum(stats.minc, minc)
            stats.maxc = np.maximum(stats.maxc, maxc)
        timeseries.append(tdata)
        stats.num_max_det = max(stats.num_max_det, len(tdata))

    class GraphCosts(gflow.StandardGraphCosts):
        def __init__(self):
            super().__init__(
                penter=args.penter,
                pexit=args.pexit,
                beta=args.fp_rate,
                max_obs_time=len(timeseries) - 1,
            )

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            """Log-probability of pairing xi(t-1) with xj(t).
            Modelled by intersection over union downweighted by an
            exponential decreasing probability on the time-difference.
            """
            iou_logprob = np.log(boxiou(x.obs, y.obs) + 1e-8)
            tdiff = y.time_index - x.time_index
            tlogprob = scipy.stats.expon.logpdf(
                tdiff, loc=1.0, scale=1 / args.exp_lambda
            )
            if x.obs.reid is not None and y.obs.reid is not None:
                reidlogprob = (
                    scipy.stats.multivariate_normal.logpdf(
                        x.obs.reid,
                        mean=y.obs.reid,
                        cov=np.diag([10 ** 2, 10 ** 2]),
                    )
                    / 5
                )
                print(reidlogprob, iou_logprob)
            else:
                reidlogprob = 0.0

            return -(iou_logprob + reidlogprob + tlogprob)

    flow = gflow.GlobalFlowMOT(
        obs=timeseries,
        costs=GraphCosts(),
        num_skip_layers=args.skip_layers,
    )

    # Solve the problem
    flowdict, _, _ = flow.solve((1, args.max_instances))
    traj = gflow.find_trajectories(flow, flowdict)
    obs_to_traj = gflow.label_observations(timeseries, traj)
    traj_info = [
        {"idx": tidx, "start": fnames[t[0].time_index], "end": fnames[t[-1].time_index]}
        for tidx, t in enumerate(traj)
    ]
    # Use filenames instead of time indices
    obs_to_traj = {fname: ids for fname, ids in zip(fnames, obs_to_traj)}
    return obs_to_traj, traj_info, stats


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser("Determine trajectories from global flow.")
    parser.add_argument(
        "-skeleton",
        type=Path,
        help="Skeleton definition",
        default=DATA_DIR / "coco.json",
    )
    parser.add_argument("--show", action="store_true", help="Show result")
    parser.add_argument("-keypoints", type=Path, help="Keypoints", required=True)
    parser.add_argument(
        "-penter", type=float, help="Probability of appearance", default=1e-2
    )
    parser.add_argument(
        "-pexit", type=float, help="Probability of appearance", default=1e-2
    )
    parser.add_argument(
        "-fp-rate", type=float, help="False positive rate of detector", default=2e-2
    )
    parser.add_argument(
        "-exp-lambda",
        type=float,
        help="Lambda of exponential distribution penalizing large time-diffs",
        default=1.0,
    )
    parser.add_argument(
        "-skip-layers",
        type=int,
        help="Number of skip-layers",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-max-instances", type=int, help="Max number of instances to expect", default=10
    )
    parser.add_argument(
        "-imagedir", type=Path, help="Optional image dir.", required=False
    )
    parser.add_argument(
        "-reidpath",
        type=Path,
        help="Use ReID features for appearance tracking.",
        required=False,
    )

    args = parser.parse_args()
    print(vars(args))
    assert args.skeleton.is_file()
    assert args.keypoints.is_file()
    assert args.imagedir is None or args.imagedir.is_dir()
    assert args.reidpath is None or args.reidpath.is_file()

    skel = json.load(open(args.skeleton, "r"))
    kpts = json.load(open(args.keypoints, "r"))

    obs_to_traj, traj_info, stats = find_trajectories(args, kpts)

    outdir: Path = (TMP_DIR / args.keypoints.stem).resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    print("Saving results to ", outdir)
    with open(outdir / "traj.json", "w") as f:

        class PathEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path):
                    return str(Path)
                return json.JSONEncoder.default(self, obj)

        f.write(
            json.dumps(
                {
                    "obs_to_traj": obs_to_traj,
                    "traj": traj_info,
                    "num_traj": len(traj_info),
                    "args": vars(args),
                },
                indent=2,
                sort_keys=False,
                cls=PathEncoder,
            )
        )

    if args.show:
        fig, (ax_orig, ax_traj) = plt.subplots(1, 2, figsize=(8, 4))
        for fname, objs in kpts.items():
            ax_orig.cla()
            ax_traj.cla()
            ax_orig.set_title("input")
            ax_traj.set_title("gflow")
            ax_orig.axis("off")
            ax_traj.axis("off")
            img = None
            if args.imagedir is not None:
                img = plt.imread(args.imagedir / fname, format="RGB")

            draw_instances(
                ax_orig,
                objs,
                np.arange(len(objs)),
                skel["limbs"],
                image=img,
                max_instances=stats.num_max_det,
            )
            draw_instances(
                ax_traj,
                objs,
                obs_to_traj[fname],
                skel["limbs"],
                image=img,
                max_instances=len(traj_info),
            )
            if img is None:
                ax_orig.set_xlim(stats.minc[0], stats.maxc[0])
                ax_orig.set_ylim(stats.minc[1], stats.maxc[1])
                ax_orig.invert_yaxis()
                ax_traj.set_xlim(stats.minc[0], stats.maxc[0])
                ax_traj.set_ylim(stats.minc[1], stats.maxc[1])
                ax_traj.invert_yaxis()
            fig.savefig(str((outdir / fname).with_suffix(".png")), bbox_inches="tight")
            plt.draw()
            plt.pause(0.001)
        plt.ioff()


if __name__ == "__main__":
    main()
