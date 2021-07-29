"""Extract re-id features for each human pose."""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from numpy.random import choice

import globalflow as gflow
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import scipy.stats
import cv2
import pickle

DATA_DIR = Path(__file__).parent / ".." / "etc" / "data"
TMP_DIR = Path(__file__).parent / ".." / "tmp"

_logger = logging.getLogger(__name__)


# def draw_skeleton(ax, xys: np.ndarray, limbs: List[Tuple[int, int]], color="k"):
#     """Draws the skeleton of a particular instance"""
#     x = xys[:, 0]
#     y = xys[:, 1]
#     for limb in limbs:
#         ax.plot(x[limb], y[limb], color=color, linewidth=1, alpha=0.5, zorder=1)
#     ax.scatter(x, y, color=color, s=8, zorder=2)


# def draw_instances(
#     ax,
#     objs: List[Dict],
#     ids: List[int],
#     limbs: List[Tuple[int, int]],
#     image: np.ndarray = None,
#     max_instances: int = 10,
#     cmap=None,
# ):
#     """Draws instance for a particular frame"""
#     if cmap is None:
#         cmap = plt.get_cmap("jet", max_instances)
#     if image is not None:
#         ax.imshow(image)
#     objs = [obj for iid, obj in zip(ids, objs) if iid > -1]
#     ids = [iid for iid in ids if iid > -1]
#     for oidx, obj in enumerate(objs):
#         xys = np.array(obj["keypoints"]).reshape(-1, 3)
#         draw_skeleton(ax, xys, limbs, color=cmap(ids[oidx]))


# @dataclass(frozen=True)
# class Detection:
#     center: np.ndarray
#     minc: np.ndarray
#     maxc: np.ndarray
#     area: float


# @dataclass
# class Stats:
#     minc: np.ndarray
#     maxc: np.ndarray
#     num_max_det: int


# def quiet_divide(a, b):
#     """Quiet divide function that does not warn about (0 / 0)."""
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         return np.true_divide(a, b)


# def boxiou(det1, det2):
#     """Computes IOU of two rectangles. Taken from
#     https://github.com/cheind/py-motmetrics/blob/6597e8a4ed398b9f14880fa76de26bc43d230836/motmetrics/distances.py#L64
#     """
#     a_min, a_max = det1.minc, det1.maxc
#     b_min, b_max = det2.minc, det2.maxc
#     # Compute intersection.
#     i_min = np.maximum(a_min, b_min)
#     i_max = np.minimum(a_max, b_max)
#     i_size = np.maximum(i_max - i_min, 0)
#     i_vol = np.prod(i_size, axis=-1)
#     # Get volume of union.
#     a_size = np.maximum(a_max - a_min, 0)
#     b_size = np.maximum(b_max - b_min, 0)
#     a_vol = np.prod(a_size, axis=-1)
#     b_vol = np.prod(b_size, axis=-1)
#     u_vol = a_vol + b_vol - i_vol
#     return np.where(
#         i_vol == 0, np.zeros_like(i_vol, dtype=float), quiet_divide(i_vol, u_vol)
#     )


# def find_trajectories(
#     args, kpts: Dict[str, Any]
# ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Stats]:
#     timeseries = []
#     fnames = []
#     stats = Stats(minc=np.array([1e3] * 2), maxc=np.array([-1e3] * 2), num_max_det=0)
#     for t, (fname, objs) in enumerate(kpts.items()):
#         fnames.append(fname)
#         tdata = []
#         for obj in objs:
#             xys = np.array(obj["keypoints"]).reshape(-1, 3)
#             minc = np.min(xys[:, :2], axis=0)
#             maxc = np.max(xys[:, :2], axis=0)
#             c = (minc + maxc) * 0.5
#             area = (maxc[0] - minc[0]) * (maxc[1] - minc[1])
#             tdata.append(Detection(c, minc, maxc, area))
#             stats.minc = np.minimum(stats.minc, minc)
#             stats.maxc = np.maximum(stats.maxc, maxc)
#         timeseries.append(tdata)
#         stats.num_max_det = max(stats.num_max_det, len(tdata))

#     class GraphCosts(gflow.StandardGraphCosts):
#         def __init__(self):
#             super().__init__(
#                 penter=args.penter,
#                 pexit=args.pexit,
#                 beta=args.fp_rate,
#                 max_obs_time=len(timeseries) - 1,
#             )

#         def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
#             """Log-probability of pairing xi(t-1) with xj(t).
#             Modelled by intersection over union downweighted by an
#             exponential decreasing probability on the time-difference.
#             """
#             iou = boxiou(x.obs, y.obs)
#             tdiff = y.time_index - x.time_index
#             logprob = np.log(iou + 1e-5) + scipy.stats.expon.logpdf(
#                 tdiff, loc=1.0, scale=1 / args.exp_lambda
#             )
#             return -logprob

#     flow = gflow.GlobalFlowMOT(
#         obs=timeseries,
#         costs=GraphCosts(),
#         num_skip_layers=args.skip_layers,
#     )

#     # Solve the problem
#     flowdict, _, _ = flow.solve((1, args.max_instances))
#     traj = gflow.find_trajectories(flow, flowdict)
#     obs_to_traj = gflow.label_observations(timeseries, traj)
#     traj_info = [
#         {"idx": tidx, "start": fnames[t[0].time_index], "end": fnames[t[-1].time_index]}
#         for tidx, t in enumerate(traj)
#     ]
#     # Use filenames instead of time indices
#     obs_to_traj = {fname: ids for fname, ids in zip(fnames, obs_to_traj)}
#     return obs_to_traj, traj_info, stats


def warp_detection(
    img: np.ndarray,
    min_c: np.ndarray,
    max_c: np.ndarray,
    size=(128, 256),
    dst: np.ndarray = None,
):
    H, W = size
    w, h = max_c - min_c
    src_coords = np.stack([min_c, min_c + [w, 0], max_c, max_c - [w, 0]], 0).astype(
        np.float32
    )
    dst_coords = np.array(
        [(0.0, 0.0), (W - 1, 0.0), (W - 1, H - 1), (0, H - 1)]
    ).astype(np.float32)
    m = cv2.getPerspectiveTransform(src_coords, dst_coords)
    return cv2.warpPerspective(img, m, (W, H), dst=dst, flags=cv2.INTER_NEAREST)


def create_warps_for_image(
    args, kpts, fname: str, skel: Dict, warp_size: Tuple[int, int] = None
) -> Optional[np.ndarray]:
    warps = []
    if warp_size is None:
        warp_size = (args.input_height, args.input_height // 2)
    path = args.imagedir / fname
    if not path.is_file():
        _logger.warning(f"{path} is not a valid file.")
        return None
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for obj in kpts[fname]:
        xys = np.array(obj["keypoints"]).reshape(-1, 3)
        minc = np.min(xys[:, :2], axis=0)
        maxc = np.max(xys[:, :2], axis=0)
        warped = warp_detection(
            img,
            minc - args.margin,
            maxc + args.margin,
            size=warp_size,
        )
        warps.append(warped)
    return warps


def extract_features(args, kpts: Dict[str, Any], skel: Dict) -> Dict[str, np.ndarray]:
    from torchreid.utils import FeatureExtractor

    warp_size = (args.input_height, args.input_height // 2)
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=str(args.osnet_weights.resolve()),
        device="cuda",
        image_size=warp_size,
    )
    # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

    reid_features = {}

    for fname, objs in kpts.items():
        warps = create_warps_for_image(args, kpts, fname, skel, warp_size)
        if warps is None:
            break
        if len(warps) > 0:
            features = extractor(warps)
            features = [f.cpu().numpy() for f in features]
        else:
            features = []
        reid_features[fname] = features  # Nx512
    return reid_features


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser("Extract ReID features for person detections")
    parser.add_argument(
        "-skeleton",
        type=Path,
        help="Skeleton definition",
        default=DATA_DIR / "coco.json",
    )
    parser.add_argument("--show", action="store_true", help="Show embedding")
    parser.add_argument(
        "--input-height", type=int, help="Input resolution for ReID", default=256
    )
    parser.add_argument("--margin", type=int, help="Margin around detection", default=0)
    parser.add_argument("-keypoints", type=Path, help="Keypoints file", required=True)
    parser.add_argument("-imagedir", type=Path, help="Image directory", required=True)
    parser.add_argument(
        "-osnet_weights", type=Path, help="Weights path for osnet", required=True
    )

    args = parser.parse_args()
    print(vars(args))
    assert args.skeleton.is_file()
    assert args.keypoints.is_file()
    assert args.osnet_weights.is_file()
    assert args.imagedir.is_dir()

    skel = json.load(open(args.skeleton, "r"))
    kpts = json.load(open(args.keypoints, "r"))

    reid_features = extract_features(args, kpts, skel)
    stem = args.keypoints.stem
    outdir: Path = (TMP_DIR / stem).resolve()
    with open(outdir / f"{stem}_reid.pkl", "wb") as f:
        f.write(pickle.dumps(reid_features))

    if args.show:
        with open(
            r"C:\dev\py-globalflow\tmp\ts18_keypoints\ts18_keypoints_reid.pkl", "rb"
        ) as f:
            reid_features = pickle.load(f)

        from sklearn.manifold import TSNE
        from sklearn import preprocessing
        from sklearn.cluster import KMeans

        fig, ax = plt.subplots()

        keys = list(reid_features.keys())
        features = []
        counts = [0]
        for k in keys:
            k_features = reid_features[k]
            features.extend(reid_features[k])
            counts.append(len(k_features))
        features = np.stack(features, 0)
        scaler = preprocessing.StandardScaler().fit(features)
        reid_features = scaler.transform(features)
        embed = TSNE(
            n_components=2,
            perplexity=50,
            early_exaggeration=5,
            learning_rate=200,
            n_iter=2000,
        ).fit_transform(features)

        choices = np.random.choice(len(keys), size=20)
        offsets = np.cumsum(counts)
        for c in choices:
            warps = create_warps_for_image(
                args, kpts, keys[c], skel, warp_size=(48, 24)
            )
            if warps is None:
                continue
            locs = [embed[offsets[c] + i] for i in range(len(warps))]
            for (w, xy) in zip(warps, locs):
                ab = AnnotationBbox(
                    OffsetImage(w), xy, frameon=True, pad=0.0, box_alignment=(0.5, 0.5)
                )
                ax.add_artist(ab)
        plt.scatter(embed[:, 0], embed[:, 1], alpha=0.5)

        # ncluster = 3
        # kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(embed)
        # for i in range(ncluster):
        #     mask = kmeans.labels_ == i
        #     plt.scatter(embed[mask, 0], embed[mask, 1], alpha=0.5)
        plt.title("TSNE plot of Re-ID features")
        plt.show()


if __name__ == "__main__":
    main()
