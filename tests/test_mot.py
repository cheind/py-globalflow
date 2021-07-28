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

    class MyCosts(gflow.StandardGraphCosts):
        def __init__(self):
            super().__init__(
                penter=1e-2, pexit=1e-2, beta=0.1, max_obs_time=len(timeseries) - 1
            )

        def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
            return -scipy.stats.norm.logpdf(y.obs, loc=x.obs + 0.1, scale=0.5)

    flow = gflow.GlobalFlowMOT(timeseries, MyCosts(), num_skip_layers=0)

    flowdict, ll, num_traj = flow.solve()
    assert_allclose(ll, 12.26, atol=1e-1)

    # [[(0, 0, u), (1, 1, u), (2, 0, u)], [(0, 1, u), (1, 3, u), (2, 2, u)]]
    trajectories = gflow.find_trajectories(flow, flowdict)
    assert len(trajectories) == 2
    assert num_traj == 2
    seq = [(n.time_index, n.obs_index) for n in trajectories[0]]
    assert seq == [(0, 0), (1, 1), (2, 0)]
    seq = [(n.time_index, n.obs_index) for n in trajectories[1]]
    assert seq == [(0, 1), (1, 3), (2, 2)]

    indices = gflow.label_observations(timeseries, trajectories)
    assert indices == [[0, 1], [-1, 0, -1, 1], [0, -1, 1]]
