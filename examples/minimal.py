import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import globalflow as gflow


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    def logp_trans(xi: gflow.FlowNode, xj: gflow.FlowNode):
        return scipy.stats.norm.logpdf(xj.obs, loc=xi.obs + 0.1, scale=0.5)

    def logp_enter(xi: gflow.FlowNode):
        return 0.0 if xi.time_index == 0 else np.log(1e-3)

    def logp_exit(xi: gflow.FlowNode):
        return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(1e-3)

    flow = gflow.GlobalFlowMOT(
        timeseries,
        logp_enter,
        logp_exit,
        logp_trans,
        gflow.default_logp_fp_fn(beta=0.05),
    )

    flowdict, ll = flow.solve()

    print(
        "optimum: log-likelihood",
        ll,
        "number of trajectories",
        len(gflow.find_trajectories(flow, flowdict)),
    ) # optimum: log-likelihood 16.76 number of trajectories 2

    plt.figure(figsize=(12, 8))
    gflow.draw.draw_graph(flow)
    plt.figure(figsize=(12, 8))
    gflow.draw.draw_flowdict(flow, flowdict)
    plt.show()

if __name__ == "__main__":
    main()
