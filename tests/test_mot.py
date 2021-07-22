import scipy.stats
import numpy as np
from numpy.testing import assert_allclose

import globalflow as gflow


def test_mot():

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

    def logp_trans(xi: gflow.FlowNode, xj: gflow.FlowNode):
        return scipy.stats.norm.logpdf(xj.obs, loc=xi.obs + 0.1, scale=0.5)

    def logp_enter(xi: gflow.FlowNode):
        return 0.0 if xi.time_index == 0 else np.log(1e-2)

    def logp_exit(xi: gflow.FlowNode):
        return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(1e-2)

    flow = gflow.GlobalFlowMOT(
        timeseries,
        logp_enter,
        logp_exit,
        logp_trans,
        gflow.default_logp_fp_fn(beta=0.1),
    )

    flowdict, ll = flow.solve()
    assert_allclose(ll, 12.26, atol=1e-1)

    # [[(0, 0, u), (1, 1, u), (2, 0, u)], [(0, 1, u), (1, 3, u), (2, 2, u)]]
    trajectories = gflow.find_trajectories(flow, flowdict)
    assert len(trajectories) == 2
    seq = [(n.time_index, n.obs_index) for n in trajectories[0]]
    assert seq == [(0, 0), (1, 1), (2, 0)]
    seq = [(n.time_index, n.obs_index) for n in trajectories[1]]
    assert seq == [(0, 1), (1, 3), (2, 2)]

    indices = gflow.label_observations(timeseries, trajectories)
    assert indices == [[0, 1], [-1, 0, -1, 1], [0, -1, 1]]
