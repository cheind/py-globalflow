import logging

import matplotlib.pyplot as plt
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

    # Define the class that provides costs.
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
    flow = gflow.GlobalFlowMOT(
        obs=timeseries,
        costs=GraphCosts(),
    )

    # Solve the problem
    flowdict, ll, num_traj = flow.solve()

    print(
        "optimum: log-likelihood", ll, "number of trajectories", num_traj
    )  # optimum: log-likelihood 6.72 number of trajectories 2

    plt.figure(figsize=(12, 8))
    gflow.draw.draw_graph(flow)
    plt.figure(figsize=(12, 8))
    gflow.draw.draw_flowdict(flow, flowdict)
    plt.show()


if __name__ == "__main__":
    main()
