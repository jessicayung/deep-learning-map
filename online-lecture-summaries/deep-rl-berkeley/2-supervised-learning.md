# L2

Lectured by Sergey Levine, Fall 2017. ([Course website](http://rail.eecs.berkeley.edu/deeprlcourse/))

(Notes are incomplete.)

## Sequential decision-making problems

- policy gives actions conditioned on observations: 
$$\pi_{\theta}(\mathbf{a_t}|\mathbf{o_t})$$
	- Fully observed setting: $$\pi_{\theta}(\mathbf{a_t}|\mathbf{s_t})$$
- Difference from IID supervised learning setting: here $$\mathbf{a_t}$$ affects $$\mathbf{o_{t+1}}$$.
- state $$\mathbf{s_t}$$
	- difference vs observations:
		- e.g. image pixel values are observations, relative positions and velocities may be state. (configuration of environment that gives rise to the observations)
		- Obs may not be sufficient to fully infer state (e.g. some objects may be obstructed from view)
		- State fully encapsulates what we need to know about the world
- Dynamics / transition distribution $$p(\mathbf{s_{t+1}}|\mathbf{s_t, a_t})$$
- Note state at timestep 3 is conditionally independent from state at timestep 1 conditional on s_2.
	- i.e. the Markov property.
	- NOT the case for observations.

## Imitation learning: supervised learning for decision making

Does direct imitation work?
	- may not work because
		- see datapoint outside of training set -> may not extrapolate properly
		- action dependent on state or previous trajectory of observations vs current obs only
		- may not have enough data

- theory: probably won't work well
	- intuition: make a small mistake bc function approx., then goes to new state it might not have seen before (or not on intended trajectory), makes a bigger mistake since not seen before
		- positive feedback from sequential nature of process, behaviour of policy diverges
- in practice: works reasonably well
	- 

How can we make it work more often?

## Case studies of recent work in (deep) imitation learning

## What's missing from imitation learning?


### Strengths and weaknesses of imitation learning algorithms
