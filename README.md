![](https://www.travis-ci.com/cheind/py-globalflow.svg?branch=main)

# **py-globalflow**
Pure Python implementation of _Global Data Association for MOT Tracking using Network Flows_ (zhang2008global) with minor tweaks.

## Example
```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import globalflow as gflow

def logp_trans(xi: gflow.FlowNode, xj: gflow.FlowNode):
    return scipy.stats.norm.logpdf(xj.obs, loc=xi.obs + 0.1, scale=0.5)

def logp_enter(xi: gflow.FlowNode):
    return 0.0 if xi.time_index == 0 else np.log(1e-3)

def logp_exit(xi: gflow.FlowNode):
    return 0.0 if xi.time_index == len(timeseries) - 1 else np.log(1e-3)

timeseries = [
    [0.0, 1.0],
    [-0.5, 0.1, 0.5, 1.1],
    [0.2, 0.6, 1.2],
]

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
```
The following graph shows the optimal trajectories

<div align="center">
<img src="etc/flow.svg" width="80%">
</div>

and problem setup

<div align="center">
<img src="etc/graph.svg" width="80%">
</div>


## Install
```bash
pip install git+https://github.com/cheind/py-globalflow
```


## References
```bibtex
@inproceedings{zhang2008global,
  title={Global data association for multi-object tracking using network flows},
  author={Zhang, Li and Li, Yuan and Nevatia, Ramakant},
  booktitle={2008 IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1--8},
  year={2008},
  organization={IEEE}
}
```
