from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import logging

_logger = logging.getLogger(__name__)


Observation = Any
"""Observation type. Can be virtually any Python object"""
ObservationTimeseries = List[List[Observation]]
"""A timeseries of observations stored as nested list of observations per timestamp."""


@dataclass(eq=True, frozen=True)
class FlowNode:
    """Represents a node in the min-cost-flow graph."""

    time_index: int  # the time index of
    obs_index: int
    tag: str
    obs: Observation = field(hash=False, compare=False)

    def __str__(self) -> str:
        return f"({self.time_index},{self.obs_index},{self.tag})"

    def __repr__(self) -> str:
        return self.__str__()

    def with_tag(self, tag: str) -> "FlowNode":
        return FlowNode(self.time_index, self.obs_index, tag, self.obs)


class GraphCosts(ABC):
    @abstractmethod
    def enter_cost(self, x: FlowNode) -> float:
        pass

    @abstractmethod
    def exit_cost(self, x: FlowNode) -> float:
        pass

    @abstractmethod
    def transition_cost(self, x: FlowNode, y: FlowNode) -> float:
        """The cost associated with linking observation x and y.
        Its guaranteed that time(x) < time(y)."""
        pass

    @abstractmethod
    def obs_cost(self, x: FlowNode) -> float:
        """The cost associated with the likelihood of observing x.
        Usually this cost is negative if the detector performance
        is better than random."""
        pass


class StandardGraphCosts(GraphCosts):
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

    def enter_cost(self, x: FlowNode) -> float:
        return 0.0 if x.time_index == 0 else -np.log(self.penter)

    def exit_cost(self, x: FlowNode) -> float:
        return 0.0 if x.time_index == self.max_obs_time else -np.log(self.pexit)

    def obs_cost(self, x: FlowNode) -> float:
        return np.log(self.beta / (1 - self.beta))


FlowDict = Dict[FlowNode, Dict[FlowNode, int]]
"""Graph edges with associated flow information."""
Trajectories = List[List[FlowNode]]
"""A list of object trajectories"""


class GlobalFlowMOT:
    """Global data association for multiple object tracking using network flows.

    This class computes a global optimal hypothesis of object trajectories
    from a set of observations.

    Kwargs
    ------
    obs: ObservationTimeseries
        List of lists of observations. Semantically a nested list at index t
        contains all observations at time t.
    costs: GraphCosts
        Instance of GraphCosts returning costs for particular graph elements.
    cost_importance_scale: float
        Conversion factor from float to integer.
    max_cost: float
        Skips all graph-edges having a cost more than the given max_cost
        value. This leads to sparser graphs and faster runtime.
    num_skip_layers: int
        The number of skip layers. If greater than zero, short-term occlusion can be handled. Defaults to zero.

    References
    ----------
    Zhang, Li, Yuan Li, and Ramakant Nevatia.
    "Global data association for multi-object tracking using network flows."
    2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2008.

    """

    START_NODE = "S"
    END_NODE = "T"

    def __init__(
        self,
        obs: ObservationTimeseries,
        costs: GraphCosts,
        cost_importance_scale: float = 1e2,
        max_cost: float = 1e4,
        num_skip_layers: int = 0,
    ):
        self.obs = obs
        self.costs = costs
        self._f2i = lambda x: int(x * cost_importance_scale)
        self._i2f = lambda x: float(x / cost_importance_scale)
        self.graph = self._build_graph(
            obs,
            costs,
            max_cost,
            num_skip_layers,
        )

    def _build_graph(
        self,
        obs: ObservationTimeseries,
        costs: GraphCosts,
        max_cost: float,
        num_skip_layers: int,
    ) -> nx.DiGraph:

        T = len(obs)
        graph = nx.DiGraph()
        # Add virtual begin and end nodes
        graph.add_node(GlobalFlowMOT.START_NODE, subset=-1)
        graph.add_node(GlobalFlowMOT.END_NODE, subset=T)

        # For each timestep...
        for tidx, tobs in enumerate(obs):
            # For each observation in timestep...
            for oidx, o in enumerate(tobs):
                u = FlowNode(tidx, oidx, "u", o)
                v = FlowNode(tidx, oidx, "v", o)
                graph.add_node(u, subset=tidx)
                graph.add_node(v, subset=tidx)

                graph.add_edge(
                    u, v, capacity=1, weight=self._f2i(costs.obs_cost(u)), color="blue"
                )

                if (cost := costs.enter_cost(u)) <= max_cost:
                    graph.add_edge(
                        GlobalFlowMOT.START_NODE,
                        u,
                        capacity=1,
                        weight=self._f2i(cost),
                        color="purple",
                    )

                if (cost := costs.exit_cost(v)) <= max_cost:
                    graph.add_edge(
                        v,
                        GlobalFlowMOT.END_NODE,
                        capacity=1,
                        weight=self._f2i(cost),
                        color="green",
                    )

                lookback_start = max(tidx - 1 - num_skip_layers, 0)

                for tprev in reversed(range(lookback_start, tidx)):
                    for pidx, p in enumerate(obs[tprev]):
                        vp = FlowNode(tprev, pidx, "v", p)
                        if (cost := costs.transition_cost(vp, u)) <= max_cost:
                            graph.add_edge(
                                vp,
                                u,
                                capacity=1,
                                weight=self._f2i(cost),
                                color="black",
                            )

        return graph

    def solve_min_cost_flow(self, num_trajectories: int) -> Tuple[FlowDict, float]:
        """Solves the MFC problem for the given number of trajectories. Returns
        the flow-dictionary and the log-likelihood of the solution."""
        assert num_trajectories > 0

        self.graph.nodes[GlobalFlowMOT.START_NODE]["demand"] = -num_trajectories
        self.graph.nodes[GlobalFlowMOT.END_NODE]["demand"] = num_trajectories

        flowdict = nx.min_cost_flow(self.graph)
        log_ll = -nx.cost_of_flow(self.graph, flowdict)
        log_ll = self._i2f(log_ll)

        return flowdict, log_ll

    def solve(
        self, bounds_num_trajectories: Tuple[int, int] = None
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
        if bounds_num_trajectories is None:
            num_obs = sum([len(tobs) for tobs in self.obs])
            bounds_num_trajectories = (1, num_obs + 1)

        opt = (None, -1e5, -1)
        for i in range(*bounds_num_trajectories):
            try:
                flowdict, ll = self.solve_min_cost_flow(i)
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
                f"Found optimimum in range {bounds_num_trajectories}, "
                f"log-likelihood {opt[1]}, number of trajectories {opt[2]}"
            )
        )
        return opt


def find_trajectories(flow: GlobalFlowMOT, flowdict: FlowDict) -> Trajectories:
    """Returns trajectories from the given flow dictionary """
    # Note, given the setup of the graph (capacity limits)
    # no edge can be shared between two trajectories. That is,
    # the number of flows through the net can be computed
    # from the number of 1s in edges from GlobalFlowMOT.START_NODE.
    def _trace(n: FlowNode):
        while n != GlobalFlowMOT.END_NODE:
            n: FlowNode
            if n.tag == "u":
                yield n
            # First non zero flow is the next node.
            n = [nn for nn, f in flowdict[n].items() if f > 0][0]

    # Find all root nodes that have positive flow from source.
    roots = [n for n, f in flowdict[GlobalFlowMOT.START_NODE].items() if f > 0]
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
