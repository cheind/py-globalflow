![](https://www.travis-ci.com/cheind/py-globalflow.svg?branch=main)

# **py-globalflow**
Pure Python implementation of _Global Data Association for MOT Tracking using Network Flows_ (zhang2008global) with minor tweaks.

## Example
```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import globalflow as gflow

timeseries = [
    [0.0, 1.0],  # obs. at t=0
    [-0.5, 0.1, 0.5, 1.1],  # obs. at t=1
    [0.2, 0.6, 1.2],  # obs. at t=2
]

def logp_trans(xi: gflow.FlowNode, xj: gflow.FlowNode):
    """Log-probability of pairing xi(t-1) with xj(t)."""
    return scipy.stats.norm.logpdf(xj.obs, loc=xi.obs + 0.1, scale=0.5)

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

## Remarks

The paper (zhang2008global) considers the problem of finding the global optimal trajectories _T_ from a given set of observerations _X_. Optimality is defined in terms maximizing the posterior probability p(_T_|_X_). Given some independence assumptions (section 3.1) the paper decomposes the distribution into two main factors: a) the likelihoods of observations p(xi|_T_) and b) the probability of a single trajectory Ti p(Ti):
- p(xi|_T_) ~ Bernoulli(1-beta)
- p(Ti) ~ Markov chain consisting of appearance, linking and disappearing probabilities between involved observations

Given probabilistic formulation, the task of finding optimal trajectories can be mapped to a min-cost-flow problem. The interpretation of this mapping is quite intuitive
> Each flow path can be interpreted as an object trajectory, the amount of the flow
sent from s to t is equal to the number of object trajectories, and the total cost of the flow on G corresponds to the loglikelihood of the association hypothesis (zhang2008global).

### p(xi|_T_)

p(xi|_T_) is modeled as a Bernoulli variable with parameter (1-b), where b(eta) is probability of being a false-positive. The derived cost term (eq. 11) Ci = log(b/(1-b)), is derived as follows ()
```
log p(xi|_T)        = log((1-bi)^fi*bi^(1-fi))
                    = fi*log(1-bi) + (1-fi)*log(bi)
                    = fi*log(1-bi) - fi*log(bi) + log(bi)
-log p(xi|_T)       = -fi*log(1-bi) + fi*log(bi) - log(bi)
                    = fi*log(bi/(1-bi)) - log(bi)
amin -log p(xi|_T)  = amin fi*log(bi/(1-bi))
                    = amin fi*ci
```
with fi being the indicator variable of whether xi is part of the solution or not. The term -log(bi) vanishes as it can be regarded constant wrt to argmin. The plot below graphs bi vs ci.

<div align="center">
<img src="etc/fpcost.svg" width="80%">
</div>

As the probability of false-positive drops below 0.5, the auxiliary edge cost between ui/vi edge cost gets negative. This allows the optimization to introduce new trajectories that increase the total flow likelihood. All other costs (pairing, appearance, disappearance) are negative log probabilities and hence positive.

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
