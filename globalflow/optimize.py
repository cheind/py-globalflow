import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
from typing import List, Tuple

from . import mot

_logger = logging.getLogger("globalflow")

TrainSequences = List[Tuple[mot.ObservationTimeseries, int]]
EStepResult = List[Tuple[mot.FlowGraph, List[mot.EdgeList]]]


@torch.no_grad()
def estep(
    train_seqs: TrainSequences,
    costs: torch.nn.Module,
    cost_scale: float,
    max_cost: float,
    num_skip_layers: int,
    traj_wnd_size: int,
) -> EStepResult:
    """Returns the E-step result for all train sequences."""
    res = []
    for obs, num_traj in train_seqs:
        fgraph = mot.build_flow_graph(
            obs,
            costs,
            cost_scale=cost_scale,
            max_cost=max_cost,
            num_skip_layers=num_skip_layers,
        )
        lower_bound = max(num_traj - traj_wnd_size, 1)
        upper_bound = min(num_traj + traj_wnd_size, fgraph.number_of_nodes() - 2)

        # Expected trajectory solution is always class 0
        flowedges = [mot.flow_edges(mot.solve_for_flow(fgraph, num_traj)[0])]
        flowedges.extend(
            [
                mot.flow_edges(mot.solve_for_flow(fgraph, t)[0])
                for t in range(lower_bound, upper_bound + 1)
                if t != num_traj
            ]
        )
        res.append((fgraph, flowedges))
    return res


def mstep_loss(
    fgraph: mot.FlowGraph, traj_edges: List[mot.EdgeList], costs: torch.nn.Module
) -> torch.Tensor:
    """Returns the mstep loss for the given flow-graph and possible trajectory candidates."""
    logits = []
    for edges in traj_edges:
        logit = torch.cat([-costs(e, fgraph.edges[e]["etype"]) for e in edges], 0).sum()
        logits.append(logit)
    logits = torch.stack(logits, 0)
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))


def optimize(
    train_seqs: TrainSequences,
    costs: torch.nn.Module,
    cost_scale: float = 100,
    max_cost: float = 1e5,
    num_skip_layers: int = 0,
    max_epochs: int = 10,
    max_msteps: int = 20,
    lr: float = 1e-1,
    traj_wnd_size: int = 1,
) -> float:
    """Optimize the parameters of graph-costs using an EM-like algorithm on a
    sequence of timeseries.

    For auto-diff purposes, we assume that costs is an instance of `torch.nn.Module`
    that defines the parameters to be optimized and otherwise follows the protocol
    of globalflow.GraphCostFn.
    """

    params = {n: p.data for n, p in costs.named_parameters() if p.requires_grad}
    _logger.info(
        f"Optimizing for parameters {params} over {len(train_seqs)} timeseries."
    )

    rel_change = lambda x, xp: abs((x - xp) / (xp + 1e-5))
    abs_change = lambda x, xp: abs(x - xp)
    stop = lambda x, xp: x > xp or rel_change(x, xp) < 1e-2 or abs_change(x, xp) < 1e-3

    for i in range(max_epochs):
        # E-Step
        estep_results = estep(
            train_seqs, costs, cost_scale, max_cost, num_skip_layers, traj_wnd_size
        )
        # M-Step
        opt = optim.SGD(
            [p for p in costs.parameters() if p.requires_grad],
            lr=lr,
            momentum=0.95,
        )
        last_loss = 1e8
        for _ in range(max_msteps):
            loss = 0.0
            for fgraph, traj_edges in estep_results:
                loss += mstep_loss(fgraph, traj_edges, costs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if stop(loss.item(), last_loss):
                break
            last_loss = loss.item()
            print(last_loss)
        print("--------------")
    params = {n: p.data for n, p in costs.named_parameters() if p.requires_grad}
    _logger.info(f"Loss {last_loss}, {params}")
    return last_loss
