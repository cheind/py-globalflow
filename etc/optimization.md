Ideas for parameter optimization in an EM-like fashion

 1. Rewrite GraphCosts to be differentiable wrt. the parameters to be optimized.
 1. Take in a list of train data, each sample a tuple of observations and expected trajectories `({xi},n)`
 1. Given a starting point for the parameters
 1. Do while total LL improves:
    1. E-step: flow-solve train sequences for the current parameters and expected trajectories. This will provide us the flow edges for the given iteration.
    2. M-step: Keeping the flow edges constant, improve the total log-likelihood (sum of ll over all train sequences) by optimization the parameters. This will require evaluating the graph-flow problem for fixed flow edges multiple times. See equations in the paper.
    