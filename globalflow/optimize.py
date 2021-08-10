import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
from typing import List, Tuple, Optional

from . import mot
from . import utils

_logger = logging.getLogger("globalflow")

LabeledSequences = List[Tuple[mot.ObservationTimeseries, int]]
EStepResult = List[Tuple[mot.FlowGraph, List[mot.EdgeList]]]


class StopCrit:
    def __init__(
        self, patience: int = 5, delta: float = 1e-2, reset_patience: bool = True
    ):
        self.patience = patience
        self.min_delta = delta
        self.best = 1e15
        self.cnt = 0
        self.reset_patience = reset_patience

    def __call__(self, loss: float):
        dlt = self.best - loss
        if self.best is None:
            self.best = loss
        elif dlt > self.min_delta:
            self.best = loss
            if self.reset_patience:
                self.cnt = 0
        elif dlt < self.min_delta:
            self.cnt += 1
            if self.cnt > self.patience:
                raise StopIteration()


@torch.no_grad()
def estep(
    train_seqs: LabeledSequences,
    costs: torch.nn.Module,
    traj_wnd_size: int,
    build_kwargs: dict,
) -> EStepResult:
    """Returns the E-step result for all train sequences."""
    res = []

    with utils.log_level(logging.ERROR):
        for obs, num_traj in train_seqs:
            fgraph = mot.build_flow_graph(obs, costs, **build_kwargs)
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
    fgraph: mot.FlowGraph,
    traj_edges: List[mot.EdgeList],
    costs: torch.nn.Module,
    mode: str = "ce",
) -> torch.Tensor:
    """Returns the mstep loss for the given flow-graph and possible trajectory candidates."""
    logits = []
    for edges in traj_edges:
        logit = torch.cat([-costs(e, fgraph.edges[e]["etype"]) for e in edges], 0).sum()
        logits.append(logit)
    logits = torch.stack(logits, 0)
    if mode == "ce":
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))
    elif mode == "hinge":
        hloss = F.multi_margin_loss(logits, torch.tensor([0]), p=1, margin=1)
        return -logits[0] + abs((logits[0] * 2e-1).item()) * hloss


@torch.no_grad()
def validate(
    val_seqs: LabeledSequences,
    costs: torch.nn.Module,
    traj_wnd_size: int,
    mstep_mode: str,
    build_kwargs: dict,
) -> torch.Tensor:
    estep_results = estep(
        val_seqs,
        costs,
        traj_wnd_size,
        build_kwargs,
    )
    loss = 0.0
    for fgraph, traj_edges in estep_results:
        loss += mstep_loss(fgraph, traj_edges, costs, mode=mstep_mode)
    return loss


def optimize(
    train_seqs: LabeledSequences,
    val_seqs: Optional[LabeledSequences],
    costs: torch.nn.Module,
    max_epochs: int = 10,
    max_msteps: int = 20,
    lr: float = 1e-2,
    traj_wnd_size: int = 1,
    mstep_mode: str = "hinge",
    build_kwargs: dict = None,
) -> float:
    """Optimize the parameters of graph-costs using an EM-like algorithm on a
    sequence of timeseries.

    For auto-diff purposes, we assume that costs is an instance of `torch.nn.Module`
    that defines the parameters to be optimized and otherwise follows the protocol
    of globalflow.GraphCostFn.
    """

    params = {n for n, p in costs.named_parameters() if p.requires_grad}
    _logger.info(
        f"Optimizing for parameters {params} over {len(train_seqs)} timeseries."
    )
    if build_kwargs is None:
        build_kwargs = {}

    stopcrit = StopCrit(patience=5, delta=1e-2, reset_patience=False)
    try:
        for eidx in range(max_epochs):
            # E-Step
            estep_results = estep(train_seqs, costs, traj_wnd_size, build_kwargs)
            # M-Step
            opt = optim.Adam(
                [p for p in costs.parameters() if p.requires_grad],
                lr=lr,
            )
            for _ in range(max_msteps):
                loss = 0.0
                for fgraph, traj_edges in estep_results:
                    loss += mstep_loss(fgraph, traj_edges, costs, mode=mstep_mode)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val_seqs is None:
                val_loss = loss.item()
            else:
                val_loss = validate(
                    val_seqs, costs, traj_wnd_size, mstep_mode, build_kwargs
                ).item()
            stopcrit(val_loss)
            _logger.debug(f"{eidx}: loss {val_loss:.2f}")
    except StopIteration:
        _logger.info("Early stopped.")
    _logger.info(f"Validation loss: {val_loss:.2f}")
    # params = {n: p.data for n, p in costs.named_parameters() if p.requires_grad}
    # _logger.info(f"Loss {last_loss}, {params}")
    # return last_loss
