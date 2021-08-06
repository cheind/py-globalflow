"""Extract re-id features for each human pose."""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

DATA_DIR = Path(__file__).parent / ".." / "etc" / "data"
TMP_DIR = Path(__file__).parent / ".." / "tmp"

_logger = logging.getLogger(__name__)


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
    dims = 0

    for fname, objs in kpts.items():
        warps = create_warps_for_image(args, kpts, fname, skel, warp_size)
        if warps is None:
            break
        if len(warps) > 0:
            features = extractor(warps)
            features = np.stack([f.cpu().numpy() for f in features], 0)
        else:
            features = []
        reid_features[fname] = features  # Nx512
        dims = features.shape[1]
    return reid_features, dims


def compress_features(
    args, reid_features: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    keys = list(reid_features.keys())
    flat_features = []
    counts = [0]
    for k in keys:
        k_features = reid_features[k]
        flat_features.extend(reid_features[k])
        counts.append(len(k_features))
    flat_features = np.stack(flat_features, 0)
    if args.compress_scale:
        scaler = preprocessing.StandardScaler().fit(flat_features)
        flat_features = scaler.transform(flat_features)

    dims = args.compress_components
    if args.compress_mode == "tsne":
        embed = TSNE(
            n_components=args.compress_components,
            perplexity=30,
            early_exaggeration=12,
            learning_rate=100,
            n_iter=2000,
        ).fit_transform(flat_features)
    elif args.compress_mode == "pca":
        pca = PCA(n_components=args.compress_components)
        embed = pca.fit_transform(flat_features)
    else:
        raise ValueError("Unknown projection mode")

    offsets = np.cumsum(counts)
    result = {
        k: embed[offsets[kidx] : offsets[kidx + 1]] for kidx, k in enumerate(keys)
    }
    return result, dims


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
    parser.add_argument(
        "--compress-mode", type=str, help="Compress ReID features", default="tsne"
    )
    parser.add_argument(
        "--compress-no-scale",
        action="store_false",
        dest="compress_scale",
        help="Disable scaling of features before compression.",
    )
    parser.add_argument(
        "--compress-components",
        type=int,
        help="Number of components compressing to",
        default=2,
    )
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

    reid_features, reid_dims = extract_features(args, kpts, skel)
    compr_features, compr_dims = compress_features(args, reid_features)
    stem = args.keypoints.stem
    outdir: Path = (TMP_DIR / stem).resolve()
    with open(outdir / f"{stem}_reid.pkl", "wb") as f:
        f.write(
            pickle.dumps(
                {
                    "reid_features": reid_features,
                    "reid_dims": reid_dims,
                    "compressed_features": compr_features,
                    "compressed_dims": compr_dims,
                }
            )
        )

    if args.show and args.compress_components == 2:
        fig, ax = plt.subplots()

        if args.compress_components == 2:
            keys = list(compr_features.keys())
            choices = np.random.choice(len(keys), size=min(100, len(keys)))
            for c in choices:
                warps = create_warps_for_image(
                    args, kpts, keys[c], skel, warp_size=(48, 24)
                )
                if warps is None:
                    continue
                locs = compr_features[keys[c]]
                for (w, xy) in zip(warps, locs):
                    ab = AnnotationBbox(
                        OffsetImage(w),
                        xy,
                        frameon=True,
                        pad=0.0,
                        box_alignment=(0.5, 0.5),
                    )
                    ax.add_artist(ab)
                plt.scatter(locs[:, 0], locs[:, 1], alpha=0.5, color="blue")

        # ncluster = 3
        # kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(embed)
        # for i in range(ncluster):
        #     mask = kmeans.labels_ == i
        #     plt.scatter(embed[mask, 0], embed[mask, 1], alpha=0.5)
        plt.title("TSNE plot of Re-ID features")
        plt.show()


if __name__ == "__main__":
    main()
