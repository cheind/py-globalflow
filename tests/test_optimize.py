from numpy.random import sample
from numpy.testing import assert_allclose

import globalflow as gflow
import logging
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.ERROR)

from globalflow.optimize import optimize


import torch
import torch.nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.distributions as dist
import torch.distributions.constraints as constraints
from collections import defaultdict


def sample_trajectory(
    t0=None,
    dur=None,
    x0=None,
    v=0.1,
    xn=0.01,
):

    if t0 is None:
        t0 = np.random.randint(0, 10)
    if dur is None:
        dur = np.random.randint(8, 20)
    if x0 is None:
        x0 = np.random.uniform(0, 100)
    ts = np.arange(dur)
    data = {t + t0: [x0 + v * t + np.random.randn() * xn] for t in ts}
    return data


def sample_trajectory_2d(
    t0=None,
    dur=None,
    x0=None,
    v=None,
    xn=0.01,
):
    if t0 is None:
        t0 = np.random.randint(0, 20)
    if dur is None:
        dur = np.random.randint(8, 20)
    if x0 is None:
        x0 = np.random.uniform(30, 60, size=(2,))
    if v is None:
        a = np.random.uniform(0, 2 * np.pi)
        v = np.array([np.cos(a), np.sin(a)])
    ts = np.arange(dur)
    data = {t + t0: [x0 + v * t + np.random.randn(2) * xn] for t in ts}
    return data


def merge_trajectories(individual_trajs, p_occ=0.01, p_fp=0.01):
    merged = defaultdict(list)
    for traj in individual_trajs:
        for t, x in traj.items():
            if np.random.uniform() > p_occ:
                merged[t].extend(x)
            if np.random.uniform() <= p_fp:
                merged[t].extend([x[0] + np.random.randn(*x[0].shape) * 1e1])

    for t, x in merged.items():
        merged[t] = torch.tensor(x)
    return merged


def plot_obs(obs, traj_labels=None, max_traj=None, ax=None):
    if ax is None:
        ax = plt.gca()
    ts = list(obs.keys())
    nc = max(ts) - min(ts) + 2
    cmap_greys = plt.get_cmap("Greys", nc)
    if traj_labels is not None:
        cmap_trajs = plt.get_cmap("jet", max_traj + 1)
    for t, xs in obs.items():
        colors = [cmap_greys(t + 1) for _ in range(len(xs))]
        if traj_labels is not None:
            for i, tl in enumerate(traj_labels[t]):
                if tl != -1:
                    colors[i] = cmap_trajs(tl)
        ax.scatter(
            xs[:, 0],
            xs[:, 1],
            color=colors,
        )


def test_optimize():

    np.random.seed(456)

    t2d = merge_trajectories(
        [
            sample_trajectory_2d(t0=0, dur=3),
            sample_trajectory_2d(t0=3, dur=5),
        ],
        p_fp=0.02,
        p_occ=0.0,
    )

    # plot_obs(t2d)
    # plt.show()

    t1 = merge_trajectories(
        [
            sample_trajectory(t0=0, x0=0.0, dur=3),
            sample_trajectory(t0=3, x0=1.0, dur=5),
        ],
        p_fp=0.02,
        p_occ=0.0,
    )
    # plot_trajectories(t1)
    # plt.show()

    t2 = merge_trajectories(
        [
            sample_trajectory(),
            sample_trajectory(),
            sample_trajectory(dur=20),
            sample_trajectory(),
        ],
        p_fp=0.02,
        p_occ=0.0,
    )
    # plot_trajectories(t2)
    # plt.show()

    t3 = merge_trajectories(
        [
            sample_trajectory(),
            sample_trajectory(),
            sample_trajectory(),
        ],
        p_fp=0.02,
        p_occ=0.0,
    )

    # plot_trajectories(t3)
    # plt.show()

    t4 = merge_trajectories(
        [
            sample_trajectory(),
            sample_trajectory(),
            sample_trajectory(),
            sample_trajectory(),
            sample_trajectory(),
            sample_trajectory(),
        ],
        p_fp=0.02,
        p_occ=0.0,
    )

    # plot_trajectories(t4)
    # plt.show()

    t5 = merge_trajectories(
        [
            sample_trajectory(),
            sample_trajectory(),
        ],
        p_fp=0.02,
        p_occ=0.0,
    )

    # plot_trajectories(t5)
    # plt.show()

    # timeseries1 = [
    #     torch.tensor([0.0, 1.0]),
    #     torch.tensor([-0.5, 0.1, 1.1]),
    #     torch.tensor([0.2, 0.6, 1.25]),
    # ]

    # timeseries2 = [
    #     torch.tensor([0.0, 1.0, 2.0]),
    #     torch.tensor([0.1, 1.0, 2.0]),
    #     torch.tensor([0.2, 1.0, -2.0]),
    #     torch.tensor([0.3, 1.0, 12.0]),
    #     torch.tensor([0.4, 1.0, 23.0]),
    #     torch.tensor([0.5, 1.0, 2.0]),
    # ]

    # timeseries3 = [
    #     torch.tensor([-1.3, 1.0, 10.3, 12.0]),
    #     torch.tensor([-1.2, 1.1, 10.4, 23.0]),
    #     torch.tensor([-1.1, 1.2, 10.5, 2.0]),
    # ]

    class TorchGraphCosts(torch.nn.Module):
        def __init__(
            self,
            penter: float = 1e-2,
            pexit: float = 1e-2,
            beta: float = 0.1,
            off: float = 0.5,
            scale: float = 0.5,
        ) -> None:
            super().__init__()
            self._zeroone = dist.transform_to(constraints.unit_interval)
            self._ho = dist.transform_to(constraints.interval(0.001, 0.999))
            self._pd = dist.transform_to(constraints.positive)
            self._upenter = Parameter(
                self._zeroone.inv(torch.tensor([penter])),
                requires_grad=True,
            )
            self._upexit = Parameter(
                self._zeroone.inv(torch.tensor([pexit])),
                requires_grad=True,
            )
            self._upbeta = Parameter(
                self._ho.inv(torch.tensor([beta])),
                requires_grad=True,
            )
            self.off = Parameter(torch.tensor([off]), requires_grad=True)
            self._uscale = Parameter(
                self._pd.inv(torch.tensor([scale])),
                requires_grad=True,
            )

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
            b = self._ho(self._upbeta)
            return torch.log(b / (1 - b))

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> torch.Tensor:
            return -dist.Normal(
                loc=y.obs - self.off, scale=self._pd(self._uscale)
            ).log_prob(x.obs)

        def constrained_params(self):
            d = {
                "penter": self._zeroone(self._upenter.detach()),
                "pexit": self._zeroone(self._upexit.detach()),
                "beta": self._ho(self._upbeta.detach()),
                "off": self.off.detach(),
                "scale": self._pd(self._uscale.detach()),
            }
            return d

    costs = TorchGraphCosts(off=0.2, scale=0.2, beta=0.02)
    # costs._upexit.requires_grad_(False)
    # costs._upenter.requires_grad_(False)
    # costs._upbeta.requires_grad_(False)

    optimize(
        train_seqs=[(t1, 2), (t2, 4), (t3, 3)],
        val_seqs=[(t4, 6), (t5, 2)],
        costs=costs,
        max_msteps=10,
        lr=1e-2,
        traj_wnd_size=1,
        max_epochs=50,
        mstep_mode="hinge",
    )
    print("-----------------------")
    print(costs.constrained_params())
    print("-----------------------")

    # assert torch.allclose(costs.off, torch.tensor([0.1]), atol=1e-2)

    fgraph = gflow.build_flow_graph(t1, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))

    fgraph = gflow.build_flow_graph(t2, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))

    fgraph = gflow.build_flow_graph(t3, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))

    fgraph = gflow.build_flow_graph(t4, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))

    fgraph = gflow.build_flow_graph(t5, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))
    # assert num_traj == 2
    # assert_allclose(ll, 12.26, atol=1e-1)
