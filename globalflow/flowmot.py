import networkx as nx
from typing import Callable, Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

Observation = Any
ObservationTimeseries = List[List[Observation]]
UnivariateLogProb = Callable[[Observation], Optional[float]]
BivariateLogProb = Callable[[Observation, Observation], Optional[float]]


def constant_prob(p: float):
    def cprob(*args, **kwargs):
        return np.log(p + 1e-5)

    return cprob


@dataclass(eq=True, frozen=True)
class FlowNode:
    time_index: int
    obs_index: int
    tag: str

    def __str__(self) -> str:
        return f"({self.time_index}, {self.obs_index}, {self.tag})"

    def __repr__(self) -> str:
        return self.__str__()


FlowDict = Dict[FlowNode, Dict[FlowNode, int]]


class GlobalFlowMOT:
    def __init__(
        self,
        obs: ObservationTimeseries,
        logp_enter_fn: UnivariateLogProb,
        logp_exit_fn: UnivariateLogProb,
        logp_trans_fn: BivariateLogProb,
        logp_tp_fn: UnivariateLogProb,
        scale: float = 1e3,
    ):
        self.obs = obs
        self._f2i = lambda x: int(x * scale)
        self._i2f = lambda x: float(x / scale)
        self.graph = self._build_graph(
            obs, logp_enter_fn, logp_exit_fn, logp_trans_fn, logp_tp_fn
        )

    def _build_graph(
        self,
        obs: ObservationTimeseries,
        logp_enter_fn: UnivariateLogProb,
        logp_exit_fn: UnivariateLogProb,
        logp_trans_fn: BivariateLogProb,
        logp_tp_fn: UnivariateLogProb,
    ) -> nx.DiGraph:

        graph = nx.DiGraph()
        # Add virtual begin and end nodes
        graph.add_node("_s_")
        graph.add_node("_t_")

        # For each timestep...
        for tidx, tobs in enumerate(obs):
            # For each observation in timestep...
            for oidx, o in enumerate(tobs):
                u = FlowNode(tidx, oidx, "u")
                v = FlowNode(tidx, oidx, "v")
                log_prob = logp_tp_fn(o)
                if log_prob is None:
                    log_prob = 0.0
                graph.add_edge(u, v, capacity=1, weight=-self._f2i(log_prob))

                log_prob = logp_enter_fn(o)
                if log_prob is not None:
                    graph.add_edge("_s_", u, capacity=1, weight=-self._f2i(log_prob))

                log_prob = logp_exit_fn(o)
                if log_prob is not None:
                    graph.add_edge(v, "_t_", capacity=1, weight=-self._f2i(log_prob))

                if tidx == 0:
                    continue

                for pidx, p in enumerate(obs[tidx - 1]):
                    log_prob = logp_trans_fn(p, o)
                    if log_prob is not None:
                        vp = FlowNode(tidx - 1, pidx, "v")
                        graph.add_edge(vp, u, capacity=1, weight=-self._f2i(log_prob))

        return graph

    def _solve(self, expected_trajectories: int = None) -> Tuple[FlowDict, float]:
        """Solves the MFC problem for the given number of trajectories. Returns an edge dict with flow information."""
        if expected_trajectories is None:
            expected_trajectories = sum([len(tobs) for tobs in self.obs])

        self.graph.nodes["_s_"]["demand"] = -expected_trajectories
        self.graph.nodes["_t_"]["demand"] = expected_trajectories

        flowdict = nx.min_cost_flow(self.graph)
        logprob = -nx.cost_of_flow(self.graph, flowdict)

        return flowdict, self._i2f(logprob)

    def _extract_flows(self, flow_dict: FlowDict) -> List[List[FlowNode]]:
        """Returns trajectories from the given flow dictionary """
        # Note, since each edge has max capacity 1, we do not need
        # to consider an edge twice in flow extraction. That is,
        # the number of flows through the net can be computed
        # from the number of 1s in edges from _s_.
        def _trace(n: FlowNode):
            while n != "_t_":
                n: FlowNode
                if n.tag == "u":
                    yield n
                # First non zero flow
                n = [nn for nn, f in flow_dict[n].items() if f > 0][0]

        roots = [n for n, f in flow_dict["_s_"].items() if f > 0]
        return [list(_trace(r)) for r in roots]


def main():
    from functools import partial
    import scipy.stats
    from pprint import pprint

    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    # timeseries = [[0.0]]

    def logp_trans(xi, xj):
        return scipy.stats.norm(xi + 0.1, 0.1).logpdf(xj)

    flow = GlobalFlowMOT(
        timeseries,
        constant_prob(0.1),
        constant_prob(0.01),
        logp_trans,
        constant_prob(1),
    )

    # [[(0, 0, u), (1, 1, u), (2, 0, u)], [(0, 1, u), (1, 3, u), (2, 2, u)]]
    flowdict, logprob = flow._solve(2)
    print(flow._extract_flows(flowdict))

    flowdict, logprob = flow._solve(1)
    print(logprob)
    flowdict, logprob = flow._solve(2)
    print(logprob)
    flowdict, logprob = flow._solve(3)
    print(logprob)
    flowdict, logprob = flow._solve(4)
    print(logprob)
    print(flow._extract_flows(flowdict))

    # print(np.exp(logprob))

    # traj, logprob = flow._solve(2)
    # print(np.exp(logprob))

    # traj, logprob = flow._solve(3)
    # print(np.exp(logprob))

    # pprint(traj, sort_dicts=False)


if __name__ == "__main__":
    main()