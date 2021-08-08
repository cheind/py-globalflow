from numpy.random import sample
from numpy.testing import assert_allclose

import globalflow as gflow
from globalflow.optimize import optimize

import torch
import torch.nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.distributions as dist
import torch.distributions.constraints as constraints


def test_constraints():

    zeroone = constraints.unit_interval
    assert dist.transform_to(zeroone)(torch.tensor([0.0])) == 0.5
    assert dist.transform_to(zeroone).inv(torch.tensor([0.5])) == 0.0


def sample_trajectory(t0=None, dur=None, v=0.1, xn=0.01, p_occ=0.01):

    if t0 is None:
        t0 = np.random.randint(0, 10)
    if dur is None:
        dur = np.random.randint(3, 10)

    x0 = np.random.uniform(0, 10)
    ts = np.arange(dur)
    bs = np.random.binomial(n=1, p=p_occ, size=dur).astype(bool)  # occ
    ts = ts[~bs]

    data = {t: [x0 + v * t + np.random.randn() * xn] for t in ts}
    return data


def test_optimize():
    timeseries1 = [
        torch.tensor([0.0, 1.0]),
        torch.tensor([-0.5, 0.1, 1.1]),
        torch.tensor([0.2, 0.6, 1.25]),
    ]

    timeseries2 = [
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([0.1, 1.0, 2.0]),
        torch.tensor([0.2, 1.0, -2.0]),
        torch.tensor([0.3, 1.0, 12.0]),
        torch.tensor([0.4, 1.0, 23.0]),
        torch.tensor([0.5, 1.0, 2.0]),
    ]

    timeseries3 = [
        torch.tensor([-1.3, 1.0, 10.3, 12.0]),
        torch.tensor([-1.2, 1.1, 10.4, 23.0]),
        torch.tensor([-1.1, 1.2, 10.5, 2.0]),
    ]

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
                self._zeroone.inv(torch.tensor([beta])),
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
            b = self._zeroone(self._upbeta)
            return torch.log(b / (1 - b))

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> torch.Tensor:
            return -dist.Normal(
                loc=y.obs - self.off, scale=self._pd(self._uscale)
            ).log_prob(x.obs)

        def constrained_params(self):
            d = {
                "penter": self._zeroone(self._upenter.detach()),
                "pexit": self._zeroone(self._upexit.detach()),
                "beta": self._zeroone(self._upbeta.detach()),
                "off": self.off.detach(),
                "scale": self._pd(self._uscale.detach()),
            }
            return d

    costs = TorchGraphCosts(off=0.2, scale=0.2, beta=0.02)
    # costs._upexit.requires_grad_(False)
    # costs._upenter.requires_grad_(False)
    costs._upbeta.requires_grad_(False)

    optimize(
        [(timeseries1, 2), (timeseries2, 1)],
        costs=costs,
        max_msteps=30,
        lr=1e-2,
        traj_wnd_size=2,
        max_epochs=20,
        mstep_mode="hinge",
    )
    print("-----------------------")
    print(costs.constrained_params())
    print("-----------------------")

    # assert torch.allclose(costs.off, torch.tensor([0.1]), atol=1e-2)

    fgraph = gflow.build_flow_graph(timeseries1, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))

    fgraph = gflow.build_flow_graph(timeseries2, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))

    fgraph = gflow.build_flow_graph(timeseries3, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))
    # assert num_traj == 2
    # assert_allclose(ll, 12.26, atol=1e-1)
