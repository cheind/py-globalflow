import numpy as np
from scipy.interpolate import interp1d
import scipy.stats
import matplotlib.pyplot as plt

from ..mot import GlobalFlowMOT, flowdict_to_trajectories


def sample_trajectory(tmax, td):
    xy_sample = np.random.uniform(0, 10, size=(4, 2))
    t_start = np.random.uniform(0, tmax / 4.0)
    t_stop = np.random.uniform(3.0 * tmax / 4, tmax)
    t_sample = np.linspace(t_start, t_stop, 4)
    fx = interp1d(t_sample, xy_sample[:, 0], kind="cubic")
    fy = interp1d(t_sample, xy_sample[:, 1], kind="cubic")

    t = np.arange(t_start, t_stop, td)
    x = fx(t)
    y = fy(t)

    tbeg = int(np.ceil(t_start / td))
    return tbeg, x, y


def sample_objects(n, tmax, td):
    t = np.arange(0, tmax, td)
    data = np.full((len(t) + 1, n, 2), np.nan)
    for i in range(n):
        tbeg, x, y = sample_trajectory(tmax, td)
        N = len(x)
        data[tbeg : tbeg + N, i, 0] = x
        data[tbeg : tbeg + N, i, 1] = y
    return t, data


def main():
    # zoom = 5
    # img = np.ones((100 * zoom, 100 * zoom, 3), dtype=np.uint8) * 255
    np.random.seed(123)
    ts, gts = sample_objects(2, 10, 0.5)
    dets = gts + np.random.randn(*gts.shape) * 0.5

    def trans(x, y):
        if np.isnan(x.obs[0]) or np.isnan(y.obs[1]):
            return np.log(1e-8)
        else:
            logx = scipy.stats.norm.logpdf(y.obs[0], loc=x.obs[0], scale=3.0)
            logy = scipy.stats.norm.logpdf(y.obs[1], loc=x.obs[1], scale=3.0)
            print(x.obs, y.obs, logx, logy)
            return logx + logy

    def logp_enter(xi):
        return 0.0 if xi.time_index == 0 else -500

    def logp_exit(xi):
        return 0.0 if xi.time_index == len(dets) - 1 else -500

    flow = GlobalFlowMOT(
        dets,
        logp_enter_fn=logp_enter,
        logp_exit_fn=logp_exit,
        logp_trans_fn=lambda x, y: trans(x, y),
        logp_tp_fn=lambda x: np.log(0.99),
        logprob_cutoff=-1e5,
    )
    flowdict, ll = flow.solve((1, 8))
    print(flowdict_to_trajectories(flow, flowdict))
    from ..draw import draw_graph, draw_flowdict

    draw_graph(flow)
    plt.figure()
    draw_flowdict(flow, flowdict)
    plt.show()

    # fig, ax = plt.subplots()
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)
    # for now, xy_gt, xy_dt in zip(ts, gts, dets):
    #     ax.scatter(xy_gt[:, 0], xy_gt[:, 1], c="k", s=1)
    #     fig.canvas.draw()
    #     plt.pause(0.033)


if __name__ == "__main__":
    main()