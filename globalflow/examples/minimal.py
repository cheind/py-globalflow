import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from globalflow.mot import FlowNode, GlobalFlowMOT, default_logp_fp_fn
from globalflow.draw import draw_flowdict, draw_graph


def main():

    timeseries = [
        [0.0, 1.0],
        [-0.5, 0.1, 0.5, 1.1],
        [0.2, 0.6, 1.2],
    ]

    def logp_trans(xi: FlowNode, xj: FlowNode):
        return scipy.stats.norm.logpdf(xj.obs, loc=xi.obs + 0.1, scale=0.1)

    def logp_enter(xi: FlowNode):
        return 0.0 if xi.time_index == 0 else np.log(0.1)

    def logp_exit(xi: FlowNode):
        return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(0.1)

    flow = GlobalFlowMOT(
        timeseries, logp_enter, logp_exit, logp_trans, default_logp_fp_fn(beta=0.01)
    )

    flowdict1, ll1 = flow.solve_min_cost_flow(1)
    flowdict2, ll2 = flow.solve_min_cost_flow(2)
    flowdict2, ll2 = flow.solve_min_cost_flow(3)

    print(ll1, ll2)
    plt.figure(figsize=(12, 8))
    draw_graph(flow)

    plt.figure(figsize=(12, 8))
    draw_flowdict(flow, flowdict1)
    plt.figure(figsize=(12, 8))
    draw_flowdict(flow, flowdict2)

    plt.show()


if __name__ == "__main__":
    main()
