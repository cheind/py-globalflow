from numpy.testing import assert_allclose

import globalflow as gflow
import torch
import torch.nn
from torch.nn.parameter import Parameter
import torch.distributions as dist


class TorchGraphCosts(gflow.GraphCosts, torch.nn.Module):
    # See https://github.com/pytorch/pytorch/blob/master/torch/distributions/constraint_registry.py
    # https://github.com/pytorch/pytorch/blob/ddd916c21010c69b6015edd419d62f28664373a7/torch/distributions/constraint_registry.py#L79
    def __init__(
        self,
        penter: float = 1e-2,
        pexit: float = 1e-2,
        beta: float = 0.1,
        max_obs_time: int = 2,
    ) -> None:
        super().__init__()
        self.logpenter = Parameter(
            torch.tensor([torch.log(penter)]),
            requires_grad=True,
        )
        self.logpexit = Parameter(
            torch.tensor([torch.log(pexit)]),
            requires_grad=True,
        )
        self.logbeta = Parameter(
            torch.tensor([torch.log(beta)]),
            requires_grad=True,
        )

        self.max_obs_time = max_obs_time

    def enter_cost(self, x: gflow.FlowNode) -> float:
        return 0.0 if x.time_index == 0 else -self.logpenter

    def exit_cost(self, x: gflow.FlowNode) -> float:
        return 0.0 if x.time_index == self.max_obs_time else -self.logpexit

    def obs_cost(self, x: gflow.FlowNode) -> float:
        return torch.log(self.logbeta.exp() / (1 - self.logbeta.exp()))

    def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
        return -dist.Normal(torch.tensor([x.obs + 0.1]), torch.tensor([0.5])).log_prob(
            torch.tensor([y.obs])
        )


def test_torch_costs():

    # should behave test_solve, except that we use torch.distributions and
    # return scalar tensors

    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    class TorchGraphCosts(gflow.GraphCosts):
        def __init__(
            self,
            penter: float = 1e-2,
            pexit: float = 1e-2,
            beta: float = 0.1,
            max_obs_time: int = 2,
        ) -> None:
            self.penter = torch.tensor([penter])
            self.pexit = torch.tensor([pexit])
            self.beta = torch.tensor([beta])
            self.max_obs_time = max_obs_time
            super().__init__()

        def enter_cost(self, x: gflow.FlowNode) -> float:
            return 0.0 if x.time_index == 0 else -torch.log(self.penter)

        def exit_cost(self, x: gflow.FlowNode) -> float:
            return 0.0 if x.time_index == self.max_obs_time else -torch.log(self.pexit)

        def obs_cost(self, x: gflow.FlowNode) -> float:
            return torch.log(self.beta / (1 - self.beta))

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return -dist.Normal(
                torch.tensor([x.obs + 0.1]), torch.tensor([0.5])
            ).log_prob(torch.tensor([y.obs]))

    fgraph = gflow.build_flow_graph(timeseries, TorchGraphCosts())
    flowdict, ll, num_traj = gflow.solve(fgraph)
    assert_allclose(ll, 12.26, atol=1e-1)
    assert num_traj == 2
