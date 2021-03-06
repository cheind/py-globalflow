import logging

import matplotlib.pyplot as plt
from networkx.classes.graph import Graph
import numpy as np
import scipy.stats

import globalflow as gflow


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    timeseries = [
        [0.0, 1.0],  # obs. at t=0
        [-0.5, 0.1, 0.5, 1.1],  # obs. at t=1
        [0.2, 0.6, 1.2],  # obs. at t=2
    ]

    # Define the class that provides costs. Here we inherit from
    # gflow.StandardGraphCosts which already predefines some costs
    # based on equations found in the paper.
    class GraphCosts(gflow.StandardGraphCosts):
        def __init__(self) -> None:
            super().__init__(
                penter=1e-3, pexit=1e-3, beta=0.05, max_obs_time=len(timeseries) - 1
            )

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            tdiff = y.time_index - x.time_index
            logprob = scipy.stats.norm.logpdf(
                y.obs, loc=x.obs + 0.1 * tdiff, scale=0.5
            ) + np.log(0.1)
            return -logprob

    # Setup the graph
    flowgraph = gflow.build_flow_graph(timeseries, GraphCosts())

    # Solve the problem globally
    flowdict, ll, num_traj = gflow.solve(flowgraph)

    print(
        "optimum: log-likelihood", ll, "number of trajectories", num_traj
    )  # optimum: log-likelihood 6.72 number of trajectories 2

    # Note, while the flowdict structure is the most general representation
    # for the solution of the problem, it might be hard to work with in
    # practice. See gflow.find_trajectories and gflow.label_observations
    # to convert between representations.

    # Plot the graph and the result.

    plt.figure(figsize=(12, 8))
    gflow.draw.draw_graph(flowgraph)
    plt.figure(figsize=(12, 8))
    gflow.draw.draw_flowdict(flowgraph, flowdict)
    plt.show()


if __name__ == "__main__":
    main()
