import argparse
import json
import logging
import pickle
import warnings
import torch
import torch.nn
import torchvision.ops as ops
import torch.distributions as dist
import torch.distributions.constraints as constraints
from torch.nn.parameter import Parameter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import globalflow as gflow
from globalflow.optimize import optimize
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

DATA_DIR = Path(__file__).parent / ".." / "etc" / "data"
TMP_DIR = Path(__file__).parent / ".." / "tmp"


@dataclass(frozen=True)
class Detection:
    box: torch.Tensor
    reid: Optional[torch.Tensor] = None


def create_observations(
    kpts_path: Path, reid_path: Optional[Path]
) -> List[List[Detection]]:
    kpts = json.load(open(kpts_path, "r"))
    reids = None
    if reid_path is not None:
        with open(reid_path, "rb") as f:
            data = pickle.load(f)
        reids = data["compressed_features"]

    timeseries = []
    for t, (fname, objs) in enumerate(kpts.items()):
        tdata = []
        if reids is not None:
            reid_features = torch.tensor(reids[fname])
        else:
            reid_features = [None] * len(objs)
        for oidx, obj in enumerate(objs):
            xys = torch.tensor(obj["keypoints"]).view(-1, 3)
            minc = xys[:, :2].min(0)[0]
            maxc = xys[:, :2].max(0)[0]
            box = torch.cat((minc, maxc))
            tdata.append(Detection(box, reid_features[oidx]))
        timeseries.append(tdata)
    return timeseries


class TorchGraphCosts(torch.nn.Module):
    def __init__(
        self,
        penter: float = 1e-2,
        pexit: float = 1e-2,
        beta: float = 0.1,
        rate: float = 1.0,
        off: float = 0.5,
        reid_vars: torch.Tensor = torch.tensor([1e2, 1e2]),
    ) -> None:
        super().__init__()
        self._zeroone = dist.transform_to(constraints.unit_interval)
        self._gt = dist.transform_to(constraints.greater_than(0.0))
        self._upenter = Parameter(
            self._zeroone.inv(torch.tensor([penter])),
            requires_grad=True,
        )
        self._upexit = Parameter(
            self._zeroone.inv(torch.tensor([pexit])),
            requires_grad=True,
        )
        self._upbeta = Parameter(
            self._zeroone.inv(torch.tensor([beta])),
            requires_grad=False,
        )
        self._urate = Parameter(self._gt.inv(torch.tensor([rate])), requires_grad=True)
        self._ureidvars = Parameter(self._gt.inv(reid_vars), requires_grad=True)

    def forward(self, e: gflow.Edge, et: gflow.EdgeType):
        x, y = e
        if et == gflow.EdgeType.ENTER:
            return self.enter_cost(y)
        elif et == gflow.EdgeType.EXIT:
            return self.exit_cost(x)
        elif et == gflow.EdgeType.OBS:
            return self.obs_cost()
        elif et == gflow.EdgeType.TRANSITION:
            return self.transition_cost(x, y)

    def enter_cost(self, x: gflow.FlowNode) -> torch.Tensor:
        return -torch.log(self._zeroone(self._upenter))

    def exit_cost(self, x: gflow.FlowNode) -> torch.Tensor:
        return -torch.log(self._zeroone(self._upexit))

    def obs_cost(self) -> torch.Tensor:
        b = self._zeroone(self._upbeta)
        return torch.log(b / (1 - b))

    def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> torch.Tensor:
        # Log-prob of bbox iou
        iou = ops.generalized_box_iou(x.obs.box.unsqueeze(0), y.obs.box.unsqueeze(0))[
            0, 0
        ]
        iou_logprob = torch.log(iou + 1e-16)

        # Log-prob of time difference
        tdiff = y.time_index - x.time_index
        t_logprob = dist.Exponential(rate=self._gt(self._urate)).log_prob(
            torch.tensor([tdiff])
        )

        # Log-prob of re-id
        if x.obs.reid is not None and y.obs.reid is not None:
            reidlogprob = dist.MultivariateNormal(
                loc=y.obs.reid, covariance_matrix=torch.diag(self._gt(self._ureidvars))
            ).log_prob(x.obs.reid)
        else:
            reidlogprob = 0.0

        return -(iou_logprob + t_logprob + +reidlogprob)

    def constrained_params(self):
        d = {
            "penter": self._zeroone(self._upenter),
            "pexit": self._zeroone(self._upexit),
            "beta": self._zeroone(self._upbeta),
            "rate": self._gt(self._urate),
            "reidvars": self._gt(self._ureidvars),
        }
        return d


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser("Determine trajectories from global flow.")
    parser.add_argument(
        "-keypoints", type=Path, help="Keypoints", required=True, nargs="+"
    )
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
        "-max-instances",
        type=int,
        help="Instances to expect.",
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "-reid",
        type=Path,
        help="Use ReID features for appearance tracking.",
        required=False,
        nargs="+",
    )

    args = parser.parse_args()
    if args.max_instances is None:
        args.max_instances = [10] * len(args.keypoints)
    if args.reid is None:
        args.reid = [None] * len(args.keypoints)

    seq = []
    for kpath, rpath in zip(args.keypoints, args.reid):
        seq.append(create_observations(kpath, rpath))

    costs = TorchGraphCosts()
    # costs._upenter.requires_grad_(False)
    costs._upexit.requires_grad_(False)
    print(costs.constrained_params())
    optimize(
        seq,
        # [(1, i + 1) for i in args.max_instances]
        [(3, 4), (5, 6)],
        costs,
        cost_scale=1e3,
        num_skip_layers=5,
        lr=1e-2,
    )
    print(costs.constrained_params())


if __name__ == "__main__":
    main()
