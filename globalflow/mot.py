from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np


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
        return f"({self.time_index}, {self.obs_index}, {self.tag})"

    def __repr__(self) -> str:
        return self.__str__()

    def with_tag(self, tag: str) -> "FlowNode":
        return FlowNode(self.time_index, self.obs_index, tag, self.obs)


FlowDict = Dict[FlowNode, Dict[FlowNode, int]]
"""Graph edges with associated flow information."""
UnivariateLogProb = Callable[[FlowNode], float]
"""A callable function returning the log-probability `log(p(xi))` of a flow-node."""
BivariateLogProb = Callable[[FlowNode, FlowNode], float]
"""A callable function return the conditional conditional log-probability log(p(xi|xj)) of two flow-nodes."""
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
    logp_enter_fn: UnivariateLogProb
        Computes the log-probability of a particular observation to appear.
    logp_exit_fn: UnivariateLogProb
        Computes the log-probability of a particular observation to disappear.
    logp_trans_fn: BivariateLogProb
        Computes the conditional log-probability of linking xi at time t-1 to
        xj at time t.
    logp_tp_fn: UnivariateLogProb
        Computes the log-probability of a particular observation to be a true-positive.
    logprob_importance_scale: float
        Conversion factor from log-prob to integer only.
    logprob_cutoff: float
        Skips all graph-edges having a probability less than the given log-prob
        value. This leads to sparser graphs and faster runtime.

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
        logp_enter_fn: UnivariateLogProb,
        logp_exit_fn: UnivariateLogProb,
        logp_trans_fn: BivariateLogProb,
        logp_tp_fn: UnivariateLogProb,
        logprob_importance_scale: float = 1e2,
        logprob_cutoff: float = np.log(1e-5),
    ):
        self.obs = obs
        self._f2i = lambda x: int(x * logprob_importance_scale)
        self._i2f = lambda x: float(x / logprob_importance_scale)
        self.graph = self._build_graph(
            obs, logp_enter_fn, logp_exit_fn, logp_trans_fn, logp_tp_fn, logprob_cutoff
        )

    def _build_graph(
        self,
        obs: ObservationTimeseries,
        logp_enter_fn: UnivariateLogProb,
        logp_exit_fn: UnivariateLogProb,
        logp_trans_fn: BivariateLogProb,
        logp_tp_fn: UnivariateLogProb,
        logprob_cutoff: float,
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

                if (log_prob := logp_tp_fn(u)) > logprob_cutoff:
                    graph.add_edge(
                        u, v, capacity=1, weight=-self._f2i(log_prob), color="blue"
                    )

                if (log_prob := logp_enter_fn(u)) > logprob_cutoff:
                    graph.add_edge(
                        GlobalFlowMOT.START_NODE,
                        u,
                        capacity=1,
                        weight=-self._f2i(log_prob),
                        color="purple",
                    )

                if (log_prob := logp_exit_fn(v)) > logprob_cutoff:
                    graph.add_edge(
                        v,
                        GlobalFlowMOT.END_NODE,
                        capacity=1,
                        weight=-self._f2i(log_prob),
                        color="green",
                    )

                if tidx == 0:
                    continue

                for pidx, p in enumerate(obs[tidx - 1]):
                    vp = FlowNode(tidx - 1, pidx, "v", p)
                    if (log_prob := logp_trans_fn(vp, u)) > logprob_cutoff:
                        graph.add_edge(
                            vp,
                            u,
                            capacity=1,
                            weight=-self._f2i(log_prob),
                            color="black",
                        )

        return graph

    def _solve(self, num_trajectories: int) -> Tuple[FlowDict, float]:
        """Solves the MFC problem for the given number of trajectories. Returns
        the flow-dictionary and the log-likelihood of the solution."""
        assert num_trajectories > 0

        self.graph.nodes[GlobalFlowMOT.START_NODE]["demand"] = -num_trajectories
        self.graph.nodes[GlobalFlowMOT.END_NODE]["demand"] = num_trajectories

        flowdict = nx.min_cost_flow(self.graph)
        log_ll = -nx.cost_of_flow(self.graph, flowdict)
        log_ll = self._i2f(log_ll)

        return flowdict, log_ll

    def _extract_flows(self, flow_dict: FlowDict) -> Trajectories:
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
                n = [nn for nn, f in flow_dict[n].items() if f > 0][0]

        # Find all root nodes that have positive flow from source.
        roots = [n for n, f in flow_dict[GlobalFlowMOT.START_NODE].items() if f > 0]
        # Trace the flow of each until termination node.
        return [list(_trace(r)) for r in roots]

    def solve(
        self, bounds_num_trajectories: Tuple[int, int] = None
    ) -> Tuple[Trajectories, float]:
        """Solves the min-cost-flow problem and returns the optimal solution.

        Args
        ----
        bounds_num_trajectories:
            Optional lower and upper bounds on number of trajectories to
            solve the min-cost-flow problem for. If not given, auto-computes
            the range.

        Returns
        -------
        trajectories: Trajectories
            The list of object trajectories
        log-likelihood: float
            The log-likelihood of the optimal solution
        """
        if bounds_num_trajectories is None:
            num_obs = sum([len(tobs) for tobs in self.obs])
            bounds_num_trajectories = (1, num_obs + 1)

        opt = (None, -1e5)
        for i in range(*bounds_num_trajectories):
            try:
                flowdict, ll = self._solve(i)
                if ll > opt[1]:
                    opt = (flowdict, ll)
            except (nx.NetworkXUnfeasible, nx.NetworkXUnbounded) as e:
                del e

        if opt[0] is None:
            raise ValueError("Failed to solve.")
        return self._extract_flows(opt[0]), opt[1]


def main():
    import scipy.stats

    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    def constant_prob(p: float):
        if p < 1e-5:
            p = 1e-5

        def cprob(*args, **kwargs):
            return np.log(p)

        return cprob

    # timeseries = [[0.0]]

    def logp_trans(xi: FlowNode, xj: FlowNode):
        return scipy.stats.norm.logpdf(xj.obs, loc=xi.obs + 0.1, scale=0.1)

    def logp_enter(xi: FlowNode):
        return 0.0 if xi.time_index == 0 else np.log(0.1)

    def logp_exit(xi: FlowNode):
        return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(0.1)

    flow = GlobalFlowMOT(
        timeseries, logp_enter, logp_exit, logp_trans, constant_prob(1)
    )

    # [[(0, 0, u), (1, 1, u), (2, 0, u)], [(0, 1, u), (1, 3, u), (2, 2, u)]]
    print(flow.solve())

    import matplotlib.pyplot as plt

    from .draw import draw_graph, draw_trajectories

    draw_trajectories(flow, flow.solve()[0])
    # draw_graph(flow)

    plt.show()


if __name__ == "__main__":
    main()
