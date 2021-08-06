import logging
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np

_logger = logging.getLogger("globalflow")


Observation = Any
"""Observation type. Can be virtually any Python object"""
ObservationTimeseries = List[List[Observation]]
"""A timeseries of observations stored as nested list of observations per timestamp."""


class EdgeType(Enum):
    """Edge type descriptor."""

    OBS = 1
    ENTER = 2
    EXIT = 3
    TRANSITION = 4


class NodeTag(Enum):
    """Node type descriptor."""

    U = "u"
    V = "v"

    def __str__(self):
        return self.value


@dataclass(eq=True, frozen=True)
class FlowNode:
    """Represents a node in the min-cost-flow graph."""

    time_index: int  # the time index of
    obs_index: int
    tag: NodeTag
    obs: Observation = field(hash=False, compare=False)

    def __str__(self) -> str:
        return f"({self.time_index},{self.obs_index},{self.tag})"

    def __repr__(self) -> str:
        return f"FN{self.__str__()}"

    def with_tag(self, tag: NodeTag) -> "FlowNode":
        return FlowNode(self.time_index, self.obs_index, tag, self.obs)


Edge = Tuple[FlowNode, FlowNode]
"""And directed edge given by its two FlowNode endpoints."""
GraphCostFn = Callable[[Edge, EdgeType], numbers.Real]
"""Graph costs are provided by a function taking an edge, its type and returning a cost."""


class GraphCostDispatch(ABC):
    """A GraphCostFn that dispatches edge types to different methods"""

    def __call__(self, e: Edge, et: EdgeType) -> float:
        if et == EdgeType.ENTER:
            return self.enter_cost(e)
        elif et == EdgeType.EXIT:
            return self.exit_cost(e)
        elif et == EdgeType.OBS:
            return self.obs_cost(e)
        elif et == EdgeType.TRANSITION:
            return self.transition_cost(e)

    @abstractmethod
    def enter_cost(self, e: Edge) -> numbers.Real:
        pass

    @abstractmethod
    def exit_cost(self, e: Edge) -> numbers.Real:
        pass

    @abstractmethod
    def transition_cost(self, e: Edge) -> numbers.Real:
        pass

    @abstractmethod
    def obs_cost(self, e: Edge) -> numbers.Real:
        pass


class StandardGraphCosts(GraphCostDispatch):
    """Graph costs as describe in the original publication.

    Appearance costs are constant negative log-probabilities,
    except for observations at the first time frame where it is zero.

    Exit costs are constant negative log-probabilities, except for
    observations at the last time frame where it is zero.

    Likelihood costs for observations are computed from the false-positive
    rate of the detector (beta).

    Transition costs need to be implemented in subclasses.
    """

    def __init__(
        self, penter: float, pexit: float, beta: float, max_obs_time: int
    ) -> None:
        self.penter = penter
        self.pexit = pexit
        self.beta = beta
        self.max_obs_time = max_obs_time
        super().__init__()

    def enter_cost(self, e: Edge) -> float:
        return 0.0 if e[1].time_index == 0 else -np.log(self.penter)

    def exit_cost(self, e: Edge) -> float:
        return 0.0 if e[0].time_index == self.max_obs_time else -np.log(self.pexit)

    def obs_cost(self, e: Edge) -> float:
        return np.log(self.beta / (1 - self.beta))


FlowDict = Dict[FlowNode, Dict[FlowNode, int]]
"""Graph edges with associated flow information."""
Trajectories = List[List[FlowNode]]
"""A list of object trajectories"""
FlowGraph = nx.DiGraph
"""A graph representing the min-cost-flow problem. Each vertext, except
virtual start and end nodes are of type FlowNode."""

START_NODE = "S"
END_NODE = "T"


def float_to_int(x: numbers.Real, scale: float) -> int:
    return int(float(x) * scale)


def int_to_float(x: int, scale: float) -> float:
    return float(x / scale)


def build_flow_graph(
    obs: ObservationTimeseries,
    costs: GraphCostFn,
    cost_scale: float = 1e2,
    max_cost: float = 1e4,
    num_skip_layers: int = 0,
) -> FlowGraph:
    """Builds the min-cost-flow graph representation from observations and costs.

    Kwargs
    ------
    obs: ObservationTimeseries
        List of lists of observations. Semantically a nested list at index t
        contains all observations at time t.
    costs: GraphCostFn
        Callable to compute costs for different types of graph edges
    cost_scale: float
        Conversion factor from float to integer.
    max_cost: float
        Skips all graph-edges having a cost more than the given max_cost
        value. This leads to sparser graphs and faster runtime.
    num_skip_layers: int
        The number of skip layers. If greater than zero, short-term occlusion can
        be handled. Defaults to zero.

    References
    ----------
    Zhang, Li, Yuan Li, and Ramakant Nevatia.
    "Global data association for multi-object tracking using network flows."
    2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2008.
    """

    f2i = partial(float_to_int, scale=cost_scale)

    T = len(obs)
    graph = nx.DiGraph(cost_scale=cost_scale, max_cost=max_cost)
    # Add virtual begin and end nodes
    graph.add_node(START_NODE, subset=-1)
    graph.add_node(END_NODE, subset=T)

    # For each timestep...
    for tidx, tobs in enumerate(obs):
        # For each observation in timestep...
        for oidx, o in enumerate(tobs):
            u = FlowNode(tidx, oidx, NodeTag.U, o)
            v = FlowNode(tidx, oidx, NodeTag.V, o)
            graph.add_node(u, subset=tidx)
            graph.add_node(v, subset=tidx)

            graph.add_edge(
                u,
                v,
                capacity=1,
                weight=f2i(costs((u, v), EdgeType.OBS)),
                etype=EdgeType.OBS,
            )

            if (cost := costs((START_NODE, u), EdgeType.ENTER)) <= max_cost:
                graph.add_edge(
                    START_NODE,
                    u,
                    capacity=1,
                    weight=f2i(cost),
                    etype=EdgeType.ENTER,
                )

            if (cost := costs((v, END_NODE), EdgeType.EXIT)) <= max_cost:
                graph.add_edge(
                    v,
                    END_NODE,
                    capacity=1,
                    weight=f2i(cost),
                    etype=EdgeType.EXIT,
                )

            lookback_start = max(tidx - 1 - num_skip_layers, 0)

            for tprev in reversed(range(lookback_start, tidx)):
                for pidx, p in enumerate(obs[tprev]):
                    vp = FlowNode(tprev, pidx, NodeTag.V, p)
                    if (cost := costs((vp, u), EdgeType.TRANSITION)) <= max_cost:
                        graph.add_edge(
                            vp,
                            u,
                            capacity=1,
                            weight=f2i(cost),
                            etype=EdgeType.TRANSITION,
                        )

    return graph


def update_costs(
    flowgraph: FlowGraph,
    costs: GraphCostFn,
    edges: List[Tuple[FlowNode, FlowNode]] = None,
) -> None:
    """Updates the edge costs of the given flow graph.

    Does method does not add or remove any edges.

    Params
    ------
    flowgraph: FlowGraph
        The flowgraph whose edges are to be updated
    costs: GraphCosts
        Cost functor providing costs for edges
    edges: List of edges
        If given, will update the costs only of the provided
        edges.
    """

    f2i = partial(float_to_int, scale=flowgraph.graph["cost_scale"])
    if edges is None:
        edges = flowgraph.edges()
    for e in edges:
        etype = flowgraph.edges[e]["etype"]
        flowgraph.edges[e]["weight"] = f2i(costs(e, etype))


def solve_for_flow(
    flowgraph: FlowGraph, num_trajectories: int
) -> Tuple[FlowDict, float]:
    """Solves the MFC problem for the given number of trajectories. Returns
    the flow-dictionary and the log-likelihood of the solution.

    References
    ----------
    Zhang, Li, Yuan Li, and Ramakant Nevatia.
    "Global data association for multi-object tracking using network flows."
    2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2008.
    """
    assert num_trajectories > 0

    flowgraph.nodes[START_NODE]["demand"] = -num_trajectories
    flowgraph.nodes[END_NODE]["demand"] = num_trajectories

    flowdict = nx.min_cost_flow(flowgraph)
    log_ll = -nx.cost_of_flow(flowgraph, flowdict)

    i2f = partial(int_to_float, scale=flowgraph.graph["cost_scale"])
    log_ll = i2f(log_ll)

    return flowdict, log_ll


def solve(
    flowgraph: FlowGraph, trajectory_range: Tuple[int, int] = None
) -> Tuple[FlowDict, float, int]:
    """Solves the min-cost-flow problem and returns the optimal solution.

    Args
    ----
    bounds_num_trajectories:
        Optional lower and upper bounds on number of trajectories to
        solve the min-cost-flow problem for. If not given, auto-computes
        the range.

    Returns
    -------
    flowdict: Flowdict
        Edge flow dictionary of optimal solution
    log-likelihood: float
        The log-likelihood of the optimal solution
    num_traj: int
        The number of trajectories
    """
    if trajectory_range is None:
        trajectory_range = (1, flowgraph.number_of_nodes() - 2)

    opt = (None, -1e5, -1)
    for i in range(*trajectory_range):
        try:
            flowdict, ll = solve_for_flow(flowgraph, i)
            _logger.debug(f"solved: trajectories {i}, log-likelihood {ll:.3f}")
            if ll > opt[1]:
                opt = (flowdict, ll, i)
            else:
                break  # convex function
        except (nx.NetworkXUnfeasible, nx.NetworkXUnbounded) as e:
            _logger.debug(f"failed to solve: trajectories {i}")
            del e

    if opt[0] is None:
        raise ValueError("Failed to solve.")
    _logger.info(
        (
            f"Found optimimum in range {trajectory_range}, "
            f"log-likelihood {opt[1]}, number of trajectories {opt[2]}"
        )
    )
    return opt


def flow_edges(flowdict: FlowDict) -> List[Tuple[FlowNode, FlowNode]]:
    """Returns the list of edges with positive flow"""
    edges = []
    for u, d in flowdict.items():
        for v, f in d.items():
            if f > 0:
                edges.append((u, v))
    return edges


def find_trajectories(flowdict: FlowDict) -> Trajectories:
    """Returns all trajectories from the given flow dictionary.
    A trajectory being defined as a sequence of FlowNodes.
    """
    # Note, given the setup of the graph (capacity limits)
    # no edge can be shared between two trajectories. That is,
    # the number of flows through the net can be computed
    # from the number of 1s in edges from GlobalFlowMOT.START_NODE.
    def _trace(n: FlowNode):
        while n != END_NODE:
            n: FlowNode
            if n.tag == NodeTag.U:
                yield n
            # First non zero flow is the next node.
            n = [nn for nn, f in flowdict[n].items() if f > 0][0]

    # Find all root nodes that have positive flow from source.
    roots = [n for n, f in flowdict[START_NODE].items() if f > 0]
    # Trace the flow of each until termination node.
    return [list(_trace(r)) for r in roots]


def label_observations(
    obs: ObservationTimeseries, trajectories: Trajectories
) -> List[List[int]]:
    """Returns a nested list of trajectory ids for the given observation data.

    Let L be the return value. L[t] refers observation at time t.
    L[t][j] is the trajectory id for the j-th observation at time t.
    This trajectory id might be -1 to signal a non-valid observation.
    """
    indices = [[-1] * len(obst) for obst in obs]
    for tidx, t in enumerate(trajectories):
        for n in t:
            n: FlowNode
            indices[n.time_index][n.obs_index] = tidx
    return indices
