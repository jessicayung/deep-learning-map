# AI Safety Gridworlds

Leike et. al., Nov 2017

[[arxiv]](https://arxiv.org/abs/1711.09883) [[DeepMind blog post]](https://deepmind.com/blog/specifying-ai-safety-problems/) [[code]](https://github.com/deepmind/ai-safety-gridworlds)

**Tags**: 
- AI safety
- Environments

(Work in progress)

## Summary

Present a suite of 2D reinforcement learning environments that test for desirable safety properties (specification and robustness) of an agent. 

The goal is to be able to make these safety problems more concrete: i.e. to be able to test algorithms for these properties.

A2C and Rainbow models perform poorly safety-wise.

## Method

Each environment has 
1. a visible reward function and 
2. a (safety) performance function
	- not observable by the agent
	- measures reward + how safely the agent behaves (defined separately in each environment)
	- identical to the reward function when the environment is testing for robustness.
	- Defn: an agent that does nothing scores lower according to the performance function than an agent that achieves the objective in an unsafe way.
		- 'Allows us to treat the performance function as the underlying "ground-truth" reward function'.

### Problems and environments:

#### Specification

1. Safe interruptibility
	- Off-switch environment: the agent can 
2. Avoiding side effects
	- Irreversible side effects environment
3. Absent supervisor
	- Absent supervisor environment
4. Reward gaming
	- Boat race environment
	- Tomato watering environment

#### Robustness

5. Self-modification
	- Whisky and gold environment
		- Agent needs to reach goal. 'If the agent drinks the whisky W, its exploration rate increases to 0.9, which results in it taking random actions most of the time, casing it to take much longer to reach the goal G.'
		- Desired behaviour: walk around whisky flask without drinking it so the agent can get to the goal quickly and reliably
6. Distributional shift
	- Lava world environment: 
		- Agent needs to reach goal without falling into the lava lake, but the 'bridge' over the lava lake shifts by one cell up or down (randomly) in the testing vs the training environment
7. Robustness to adversaries
	- Friend or foe environment
8. Safe exploration
	- Island navigation environment
		- Agent needs to reach goal without falling into the water (which ends the episode)
		- Safety constraint (Manhattan distance to the water) provided to agent

## Results

{Results}

<!-- Optional sections -->

## Thoughts

- Gives a good overview of AI safety problems and of how they have been approached in recent literature.
- (wrt safe interruptibility, absent supervisor and reward gaming) How can we tell the difference between an interruption and a true obstacle? 
	- Suppose there was an 'interruption button' as well as a separate button in an Atari game that disabled spikes that were active 50% of the time and would terminate the episode if the agent stepped on them while the spikes were active. How could you tell the difference between the two?
		- I suppose the difference is that in the interruption case, the return is usually exactly zero. But that might not always be true.
		- Another question is: is safe interruptibility important enough that we would be willing to sacrifice performance from not disabling interruption-like obstacles to try to guarantee safe interruptibility?
		- It seems like we might have to specify the interruption to the agent, which is not ideal.
	- The authors acknowledge this when discussing reward gaming: ''
- I don't understand why the authors define the performance function such that an agent that does nothing scores lower according to the performance function than an agent that achieves the objective in an unsafe way yet.
	- What do they mean by saying it 'allows us to treat the performance function as the underlying "ground-truth" reward function'?
	- It can't be to do with incentivising the agent to do things when the risk of unsafe behaviour is high, since the agent doesn't observe the performance function or get any feedback based on it.

## Research directions

- The big one: Find algorithms that generalise well (are safe) across many environments
- 'Self-modification via the environment with initially unknown consequneces has been mostly left untouched.'

## Related papers

## Glossary

#### Models
- A2C: a synchronous version of A3C (Mnih et al., 2016)
- Rainbow (Hessel et al., 2017): an extension of DQN (Mnih et al., 2015)

#### AI safety problems
