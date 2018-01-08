# Glossary

(Work in progress. The focus is on terms that won't be part of most introductory courses since those definitions are easy to find and are usually in the [WildML glossary](http://www.wildml.com/deep-learning-glossary/).)


### Frameworks


#### Options framework
- involves abstractions over the space of actions
- at each step, the agent chooses either a one-step 'primitive' action or a 'multi-step' action policy (option). Each option defines a policy over actions (either primitive or other options) and can be terminated according to a stochastic function of $\beta$.
- Paper: Sutton et. al.

Definition from Kulkarni and Narasimhan et. al (2016) 

- Policy gradient methods

### Training methods

- Backpropagation
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