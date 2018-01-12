# Glossary

(Work in progress. The focus is on terms that won't be part of most introductory courses since those definitions are easy to find and are usually in the [WildML glossary](http://www.wildml.com/deep-learning-glossary/).)

See also the [basics glossary](basics-glossary.md).

### Frameworks


#### Options framework
- involves abstractions over the space of actions
- at each step, the agent chooses either a one-step 'primitive' action or a 'multi-step' action policy (option). Each option defines a policy over actions (either primitive or other options) and can be terminated according to a stochastic function of $\beta$.
- Paper: Sutton et. al.

Definition from Kulkarni and Narasimhan et. al (2016) 

- Policy gradient methods

### Training methods

- [Backpropagation](basics-glossary.md)
- Synthetic gradients
	- Result: Faster updating of parameter weights
	- Method: Using predicted 'synthetic gradients' (estimate based on local gradient) instead of true backpropagated error gradients
	- [[Paper (Jaderberg et. al., Jul 2017)]](https://arxiv.org/pdf/1608.05343.pdf)

### Models

- A3C (Asynchoronous Advantage Actor-Critic)
	- Actor-critic: 
		- Two outputs: 
			1. Actor: outputs Policy, i.e. Q-values $Q(s,a_i)$ for all $a_i$, possible actions via Softmax
			2. Critic: outputs Value of state we're in $V(s)$
	- Asynchronous
		- Multiple agents tackling the same environment, each initalised differently (diff random seed)
			- More experience to learn from
			- Reduces chance of all agents being stuck in a local max
			- Can combine N nets into one single net,
				- where N = number of agents. 
				- So weights are shared.
		- Agents share experience by contributing to a common critic
	- Advantage
		- Have two losses, one for each output (Value loss, policy loss)
		- Value loss: TODO (fill in)
		- Policy loss: 
			- Let Advantage A = Q(s,a) - V(s)
				- How much better is the Q-value you're selecting compared to the 'known' V value across agents?
			- Goal is to maximise advantage: encourages actions that have Q(s,a) > V.

- A2C (Synchronous A3C: Advantage Actor-Critic)
- Rainbow
- Neural Turing Machine
- DQN
- Capsule Networks
- Dilated convolutions
	- Convolutions with filter cells that are regularly spaced out.
	- Purpose: Receptive field grows quicker, so can merge more spatial information across input (keeping filter size constant).
- Dilated LSTMs
- Skip connections

### Reinforcement Learning
- Intuition of RL: 
	- Loop through two steps:
		- Agent performs action. 
		- State may change, agent may get reward.
	- Agent explores the environment by taking actions.
	- Actions involve time
	- Don't pre-program procedures in agent, but agent knows list of actions
- Bellman Equation
	- $V(s) = \max{a}(R(s,a)+\gamma E[V(s')])$
		- where $\gamma$ is the discount factor.
		- Deterministic version: $V(s) = \max{a}(R(s,a)+\gamma V(s'))$
		- Expanded for MDPs: $V(s) = \max{a}(R(s,a)+\gamma \sum_{s'} P(s,a,s')V(s'))$
- Plans vs Policies: 
	- Plans comprise the optimal action for each state, with no stochasticity. Policies incorporate stochasticity.
- Deterministic vs non-deterministic search:
	- Deterministic search: Agent's intention maps 100% to agent's action.
	- Non-deterministic search: Small chance of agent acting differently to how it intends to act
- Markov Decision Processes (MDP)
	- Mathematical framework for modelling decision-making where outcomes are partly random and partly under the control of a decision-maker
	- Markov Property: 
		- Memorylessness: Conditional P(X) dist depends only on present state
	- Associated Bellman eqn: $V(s) = \max{a}(R(s,a)+\gamma E[V(s')])$
		- aka $V(s) = \max{a}(R(s,a)+\gamma \sum_{s'} P(s,a,s')V(s'))$
- Q-learning
	- Give values to actions $Q(s_0,a_i)$ instead of states
		- $Q(s,a) = R(s,a)+\gamma \sum_{s'} P(s,a,s')V(s')$
			- i.e. $Q(s,a) = R(s,a)+\gamma \sum_{s'} P(s,a,s')\max{a'}Q(s',a')$
- Temporal Difference
	- TODO: refine
	- (Consider Q-learning under deterministic search for convenienc)
	- $TD_t(a,s) = Q_t(s,a) - Q_{t-1}(s,a) = R(s,a)+\gamma\max{a'}Q(s',a') - Q_{t-1}(s,a)$
	- $TD(a,s)$ may be nonzero because of randomness. (Though we've *written* the deterministic search version of )
- Update eqn: $Q_t(s,a) = Q_{t-1}(s,a) + \alpha TD_t(a,s)$
	- $\alpha$ is the learning rate.
	- Hope: algorithm will converge to the 'correct' Q-value, unless the environment is constantly changing.

References:
- RL: AI A to Z course