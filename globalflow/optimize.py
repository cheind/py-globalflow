import torch
import torch.optim as optim
import logging
from typing import List, Tuple

from . import mot

_logger = logging.getLogger("globalflow")


def optimize(
    obs_sequences: List[mot.ObservationTimeseries],
    trajectory_ranges: List[Tuple[int, int]],
    costs: torch.nn.Module,
    cost_scale: float = 100,
    max_cost: float = 1e5,
    max_epochs: int = 10,
    max_msteps: int = 100,
    lr: float = 1e-1,
) -> float:
    """Optimize the parameters of graph-costs using an EM-like algorithm on a
    sequence of timeseries.

    For auto-diff purposes, we assume that costs is an instance of `torch.nn.Module`
    that defines the parameters to be optimized and otherwise follows the protocol
    of globalflow.GraphCostFn.
    """

    params = {n: p.data for n, p in costs.named_parameters() if p.requires_grad}
    _logger.info(
        f"Optimizing for parameters {params} over {len(obs_sequences)} timeseries."
    )

    max_ll = -float("inf")
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
        opt = optim.SGD(costs.parameters(), lr=lr, momentum=0.95)
        for _ in range(max_msteps):
            # Update costs
            loss = 0.0
            for edges, fgraph in zip(problem_edges, fgraphs):
                loss = (
                    loss
                    + torch.cat(
                        [costs(e, fgraph.edges[e]["etype"]) for e in edges], 0
                    ).sum()
                )
            if -loss.item() <= max_ll:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()

        if -loss.item() <= max_ll:
            break
        max_ll = -loss.item()

    params = {n: p.data for n, p in costs.named_parameters() if p.requires_grad}
    _logger.info(f"Log-Likelihood {max_ll}, {params}")
    return max_ll
