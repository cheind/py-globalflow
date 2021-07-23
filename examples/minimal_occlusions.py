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
        [0.2, 0.6],  # obs. at t=2
        [0.3, 0.6, 1.3],  # obs. at t=3
    ]

    def logp_trans(xi: gflow.FlowNode, xj: gflow.FlowNode):
        """Log-probability of pairing xi(t-1) with xj(t)."""
        tdiff = xj.time_index - xi.time_index
        return scipy.stats.norm.logpdf(
            xj.obs, loc=xi.obs + 0.1 * tdiff, scale=0.5
        ) + np.log(0.1)

    def logp_enter(xi: gflow.FlowNode):
        """Log-probability of xi(t) appearing."""
        return 0.0 if xi.time_index == 0 else np.log(1e-3)

    def logp_exit(xi: gflow.FlowNode):
        """Log-probability of xi(t) disappearing."""
        return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(1e-3)

    # Setup the graph
    flow = gflow.GlobalFlowMOT(
        timeseries,
        logp_enter,
        logp_exit,
        logp_trans,
        gflow.default_logp_fp_fn(beta=0.05),
        num_skip_layers=1,
    )

    # Solve the problem
    flowdict, ll = flow.solve()

    print(
        "optimum: log-likelihood",
        ll,
        "number of trajectories",
        len(gflow.find_trajectories(flow, flowdict)),
    )  # optimum: log-likelihood 16.76 number of trajectories 2

    plt.figure(figsize=(12, 8))
    gflow.draw.draw_graph(flow)
    plt.figure(figsize=(12, 8))
    gflow.draw.draw_flowdict(flow, flowdict)
    plt.show()


if __name__ == "__main__":
    main()
