import torch
import torch.optim as optim
from typing import List, Tuple

from . import mot


def optimize(
    obs_sequences: List[mot.ObservationTimeseries],
    trajectory_ranges: List[Tuple[int, int]],
    costs: mot.GraphCosts,
    cost_scale: float = 100,
    max_cost: float = 1e5,
    max_epochs: int = 10,
    max_msteps: int = 100,
    lr: float = 1e-3,
):
    """Optimize the parameters of graph-costs using an EM-like algorithm on a
    sequence of timeseries.

    For auto-diff purposes, we assume that costs is an instance of `torch.nn.Module`
    that defines the parameters to be optimized and otherwise follows the protocol
    of globalflow.GraphCosts.
    """
    assert isinstance(costs, torch.nn.Module)

    ll = -float("inf")
    for i in range(max_epochs):
        # E-Step
        with torch.no_grad():
            fgraphs = [
                mot.build_flow_graph(
                    obs, costs, cost_scale=cost_scale, max_cost=max_cost
                )
                for obs in obs_sequences
            ]
        esteps = [
            mot.solve(fgraph, traj_range)
            for fgraph, traj_range in zip(fgraphs, trajectory_ranges)
        ]

        # M-Step

        # Get the active-edges (i.e flow > 0). This algorithm will consider
        # only those during optimization. For each problem instance, we cthen store
        # a dictionary from edges to cost and type. It will be this structure
        # that we will invoke during the M-step for optimization.
        problem_edges = [mot.flow_edges(t[0]) for t in esteps]
        opt = optim.SGD(costs.get_parameter(), lr=lr, momentum=0.95)
        for _ in range(max_msteps):
            # Update costs
            losses = []
            for edges, fgraph in zip(problem_edges, fgraphs):
                losses.extend(
                    [mot.compute_cost_dispatch(fgraph, costs, e) for e in edges]
                )
            opt.zero_grad()
            loss = torch.sum(losses)
            loss.backward()
            opt.step()
            print(loss)
        ll = -loss
        print(ll)