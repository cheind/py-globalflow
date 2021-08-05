from numpy.testing import assert_allclose

import globalflow as gflow
from globalflow.optimize import optimize

import torch
import torch.nn
from torch.nn.parameter import Parameter
import torch.distributions as dist
import torch.distributions.constraints as constraints


def test_constraints():

    zeroone = constraints.unit_interval
    assert dist.transform_to(zeroone)(torch.tensor([0.0])) == 0.5
    assert dist.transform_to(zeroone).inv(torch.tensor([0.5])) == 0.0


def test_optimize():
    timeseries = [
        torch.tensor([0.0, 1.0]),
        torch.tensor([-0.5, 0.1, 1.1]),
        torch.tensor([0.2, 0.6, 1.25]),
    ]

    class TorchGraphCosts(gflow.GraphCosts, torch.nn.Module):
        def __init__(
            self,
            penter: float = 1e-2,
            pexit: float = 1e-2,
            beta: float = 0.1,
            off: float = 0.5,
            max_obs_time: int = 2,
        ) -> None:
            super().__init__()
            self._zeroone = dist.transform_to(constraints.unit_interval)
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

            self.max_obs_time = max_obs_time

        def enter_cost(self, x: gflow.FlowNode) -> float:
            return (
                torch.tensor([0.0])
                if x.time_index == 0
                else -torch.log(self._zeroone(self._upenter))
            )

        def exit_cost(self, x: gflow.FlowNode) -> float:
            return (
                torch.tensor([0.0])
                if x.time_index == self.max_obs_time
                else -torch.log(self._zeroone(self._upexit))
            )

        def obs_cost(self, x: gflow.FlowNode) -> float:
            b = self._zeroone(self._upbeta)
            return torch.log(b / (1 - b))

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return -dist.Normal(
                loc=y.obs - self.off, scale=torch.tensor([0.5])
            ).log_prob(x.obs)

    costs = TorchGraphCosts(off=0.2)
    costs._upexit.requires_grad_(False)
    costs._upenter.requires_grad_(False)
    costs._upbeta.requires_grad_(False)

    optimize([timeseries], [(1, 10)], costs=costs, max_msteps=100, lr=1e-2)
    assert torch.allclose(costs.off, torch.tensor([0.1]), atol=1e-2)

    fgraph = gflow.build_flow_graph(timeseries, costs)
    flowdict, ll, num_traj = gflow.solve(fgraph, (1, 10))
    assert num_traj == 2
    assert_allclose(ll, 12.26, atol=1e-1)
