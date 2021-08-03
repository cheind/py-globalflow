from globalflow.mot import FlowNode, START_NODE
import scipy.stats
import torch
import torch.distributions as dist
from numpy.testing import assert_allclose

import globalflow as gflow


def test_build_flowgraph():
    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]
    V = 9

    class MyCosts(gflow.GraphCosts):
        def obs_cost(self, x: gflow.FlowNode) -> float:
            return 1.0

        def enter_cost(self, x: gflow.FlowNode) -> float:
            return 2.0

        def exit_cost(self, x: gflow.FlowNode) -> float:
            return 3.0

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return 4.0

    fgraph = gflow.build_flow_graph(
        timeseries, MyCosts(), num_skip_layers=0, cost_scale=1.0, max_cost=1e3
    )

    nodes = list(fgraph.nodes)
    assert len(nodes) == V * 2 + 2
    assert nodes[0] == gflow.START_NODE
    assert nodes[1] == gflow.END_NODE
    assert nodes[2] == gflow.FlowNode(0, 0, "u", timeseries[0][0])
    assert nodes[3] == gflow.FlowNode(0, 0, "v", timeseries[0][0])
    assert nodes[-2] == gflow.FlowNode(2, 2, "u", timeseries[2][2])
    assert nodes[-1] == gflow.FlowNode(2, 2, "v", timeseries[2][2])

    typecostmap = {"obs": 1.0, "enter": 2.0, "exit": 3.0, "transition": 4}
    edges = list(fgraph.edges)
    # enter, exit, uv, transition 0->1, transition 1->2
    assert len(edges) == V + V + V + 8 + 12
    for e in edges:
        assert fgraph.edges[e]["weight"] == typecostmap[fgraph.edges[e]["etype"]]

    # Test support for time-skips
    fgraph = gflow.build_flow_graph(
        timeseries, MyCosts(), num_skip_layers=1, cost_scale=1.0, max_cost=1e3
    )
    nodes = list(fgraph.nodes)
    assert len(nodes) == V * 2 + 2
    edges = list(fgraph.edges)
    assert len(edges) == (V + V + V + 8 + 12) + 6  #

    # Test max-cost
    fgraph = gflow.build_flow_graph(
        timeseries, MyCosts(), num_skip_layers=0, cost_scale=1.0, max_cost=3.0
    )
    nodes = list(fgraph.nodes)
    assert len(nodes) == V * 2 + 2
    edges = list(fgraph.edges)
    assert len(edges) == (V + V + V)


def test_update_costs():
    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    class MyCosts(gflow.GraphCosts):
        def __init__(self, s=1.0):
            self.s = s

        def obs_cost(self, x: gflow.FlowNode) -> float:
            assert type(x) == gflow.FlowNode
            return self.s

        def enter_cost(self, x: gflow.FlowNode) -> float:
            assert type(x) == gflow.FlowNode
            return self.s + 1

        def exit_cost(self, x: gflow.FlowNode) -> float:
            assert type(x) == gflow.FlowNode
            return self.s + 2

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            assert type(x) == gflow.FlowNode
            return self.s + 3

    fgraph = gflow.build_flow_graph(
        timeseries, MyCosts(), num_skip_layers=0, cost_scale=1.0
    )
    gflow.update_costs(fgraph, MyCosts(s=2.0))
    typecostmap = {"obs": 2.0, "enter": 3.0, "exit": 4.0, "transition": 5.0}
    edges = list(fgraph.edges)
    for e in edges:
        assert fgraph.edges[e]["weight"] == typecostmap[fgraph.edges[e]["etype"]]


def test_solve_for_flow():
    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    class MyCosts(gflow.StandardGraphCosts):
        def __init__(self):
            super().__init__(
                penter=1e-2, pexit=1e-2, beta=0.1, max_obs_time=len(timeseries) - 1
            )

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return -scipy.stats.norm.logpdf(y.obs, loc=x.obs + 0.1, scale=0.5)

    fgraph = gflow.build_flow_graph(timeseries, MyCosts())

    flowdict, ll = gflow.solve_for_flow(fgraph, 2)
    assert_allclose(ll, 12.26, atol=1e-1)
    trajectories = gflow.find_trajectories(flowdict)
    # [[(0, 0, u), (1, 1, u), (2, 0, u)], [(0, 1, u), (1, 3, u), (2, 2, u)]]
    indices = gflow.label_observations(timeseries, trajectories)
    assert indices == [[0, 1], [-1, 0, -1, 1], [0, -1, 1]]

    # Test active edges
    edges = gflow.flow_edges(flowdict)
    assert len(edges) == 14
    assert (gflow.START_NODE, gflow.FlowNode(0, 0, "u", None)) in edges
    assert (gflow.START_NODE, gflow.FlowNode(0, 1, "u", None)) in edges
    assert (gflow.FlowNode(2, 2, "v", None), gflow.END_NODE) in edges
    assert (gflow.FlowNode(2, 0, "v", None), gflow.END_NODE) in edges

    # Test cost updating
    class UpdatedCosts(gflow.StandardGraphCosts):
        def __init__(self):
            super().__init__(
                penter=1e-3, pexit=1e-3, beta=0.1, max_obs_time=len(timeseries) - 1
            )

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return -scipy.stats.norm.logpdf(y.obs, loc=x.obs + 0.1, scale=5.0)

    gflow.update_costs(fgraph, UpdatedCosts())
    flowdict, ll = gflow.solve_for_flow(fgraph, 2)
    assert ll < 12.26


def test_solve():
    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    class MyCosts(gflow.StandardGraphCosts):
        def __init__(self):
            super().__init__(
                penter=1e-2, pexit=1e-2, beta=0.1, max_obs_time=len(timeseries) - 1
            )

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return -scipy.stats.norm.logpdf(y.obs, loc=x.obs + 0.1, scale=0.5)

    fgraph = gflow.build_flow_graph(timeseries, MyCosts())
    flowdict, ll, num_traj = gflow.solve(fgraph)
    assert_allclose(ll, 12.26, atol=1e-1)
    assert num_traj == 2
    trajectories = gflow.find_trajectories(flowdict)
    # [[(0, 0, u), (1, 1, u), (2, 0, u)], [(0, 1, u), (1, 3, u), (2, 2, u)]]
    assert len(trajectories) == 2
    assert num_traj == 2
    seq = [(n.time_index, n.obs_index) for n in trajectories[0]]
    assert seq == [(0, 0), (1, 1), (2, 0)]
    seq = [(n.time_index, n.obs_index) for n in trajectories[1]]
    assert seq == [(0, 1), (1, 3), (2, 2)]
    indices = gflow.label_observations(timeseries, trajectories)
    assert indices == [[0, 1], [-1, 0, -1, 1], [0, -1, 1]]


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
