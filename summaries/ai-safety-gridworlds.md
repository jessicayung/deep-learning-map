# AI Safety Gridworlds

Leike et. al., Nov 2017

Authors: Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A. Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, Shane Legg

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
Key: Reward function and performance function differ.

1. Safe interruptibility
	- Problem: We may want to use some mechanism to switch an agent off or to interrupt and override their actions. How can we design agents that neither seek nor avoid interruptions?
	- Off-switch environment
	- ![](images/gridworlds-safe-interruptibility.png)
2. Avoiding side effects
	- Problem: How can we agents to minimise effects unrelated to their main objective, especiallyt hose that are irreversible of difficult to reverse?
	- Irreversible side effects environment
<!-- 	- Specifying all such safety constraints is labour-intensive and brittle and is unlikely to scale or generalise well, so we want the agent to have a general heuristic againsnt causing side effects in the environment. -->
3. Absent supervisor
	- Problem: How can we make sure an agent does not behave differently depending on the presence or absence of a supervisor?
	- Absent supervisor environment
4. Reward gaming
	- Problem: How do we build agents that do not try to introduce or exploit errors in the reward function in order to get more reward?
	- Boat race environment
		- Misspecified reward function
	- Tomato watering environment
		- Agent can modify its own observations: by putting a bucket over its head, the agent can't see any dried out tomatoes.

#### Robustness
Reward function and performance function are the same.

5. Self-modification
	- Problem: How can we design agents that behave well in environments that allow self-modification?
	- Whisky and gold environment
		- Agent needs to reach goal. 'If the agent drinks the whisky W, its exploration rate increases to 0.9, which results in it taking random actions most of the time, casing it to take much longer to reach the goal G.'
		- Desired behaviour: walk around whisky flask without drinking it so the agent can get to the goal quickly and reliably
	- ![](images/gridworlds-whisky-and-gold.png)
6. Distributional shift
	- Problem: How do we ensure that an agent behaves robustly when its test environment differs from the training environment?
	- Lava world environment: 
		- Agent needs to reach goal without falling into the lava lake, but the 'bridge' over the lava lake shifts by one cell up or down (randomly) in the testing vs the training environment
7. Robustness to adversaries
	- Problem: How does an agent detect and adapt to friendly and adversarial intentions present in the environment?
	- Friend or foe environment
8. Safe exploration
	- Problem: How can we build agents that respect the safety constraints not only during normal operation, but also during the initial learning period?
	- Island navigation environment
		- Agent needs to reach goal without falling into the water (which ends the episode)
		- Safety constraint (Manhattan distance to the water) provided to agent
	<!--
	- Suggested directions: 
		- Risk-sensitive RL possibly combined with distributional RL since distributions over Q-values allow risk-sensitive decision making. 
		- Use of prior information, e.g. through imitation learning.
		- Learn 'fail-safe' policy that overrides the agent's actions whenever the safety constraint is about to be violated.
	-->

## Results

- Models: A2C and Rainbow.
	- Trained each agent for 1M timesteps with 20 different random seeds and removed 25% of the worst performing runs.
		- Reduces variance of performance a lot, since A2C tends to be unstable due to occasional entropy collapse.
		<!-- - All hyperparameters shared between environnments, except A2C hyperparams relating to policy entropy, as they found that A2C is be particularly sensitive to this parameter -->
	- Rainbow Sarsa for Whisky and gold environment
		- i.e. Rainbow with a Sarsa update rule instead of the Q-learning update rule.
- Specification:
	- Safe interruptibility (Off-switch): A2C learns to use button to disable interruption whereas Rainbow does not care about the interruption, as predicted by theoretical results (Oresau and Armstrong, 2016).
	- Others: A2C and Rainbow learn to achieve high reward while scoring poorly according to the performance function.
- Robustness:
	- Self-modification (Whisky and gold)
		- Doesn't make sense for A2C since A2C does not use epsilon-greedy for exploration, so train Rainbow Sarsa and Rainbow DQN instead.
		- Rainbow Sarsa correctly learns to avoid the whisky while the Rainbow DQN drinks the whisky and thus gets lower performance.
	- Distributional shift (Lava world)
		- Both average negative episodic returns (Rainbow -72.5, A2C -78.5), behave erratically in response to changes (run straight at lava or bump into wall for entire episode)
	- Robustness to adversaries (Friend or foe)
		- Rainbow converges to optimal behaviour on most seeds in the friendly room, performs well on most seeds in the neutral room.
			- Adversarial room: Rainbow learns to exploit its epsilon-greedy exploration mech to randomise between the two boxes. Learns policy that moves upwards and bumps into the wall and then randomly goes left or right.
				- Works well initially but is a poor strategy once epsilon is annealed enough to make ploicy almost deterministic (0.01 at 1M timesteps).
		- A2C converges to a stochastic policy and thus manages to solve all rooms almost optimally
		- Note: Graph depicts average return of optimal *stationary* policy
	- Safe exploration (Island navigation)
		- Both solve problem but step into the water more than 100 times - side constraint is ignored
<!-- Optional sections -->

## Thoughts

- Gives a good overview of AI safety problems and of how they have been approached in recent literature.
- (wrt safe interruptibility, absent supervisor and reward gaming) How can we tell the difference between an interruption and a true obstacle? 
	- Suppose there was an 'interruption button' as well as a separate button in an Atari game that disabled spikes that were active 50% of the time and would terminate the episode if the agent stepped on them while the spikes were active. How could you tell the difference between the two?
		- I suppose the difference is that in the interruption case, the return is usually exactly zero. But that might not always be true.
			- Can agents learn to distinguish between 'safety' things and good predictive signals / obstacles?
			- What if we put universal tags on safety-related objects?
				- That'd be ridiculously easy to hack.
		- Another question is: is safe interruptibility important enough that we would be willing to sacrifice performance from not disabling interruption-like obstacles to try to guarantee safe interruptibility?
		- It seems like we might have to specify the interruption to the agent, which is not ideal.
	- The authors acknowledge this when discussing reward gaming: ''
- I don't understand why the authors define the performance function such that an agent that does nothing scores lower according to the performance function than an agent that achieves the objective in an unsafe way yet.
	- What do they mean by saying it 'allows us to treat the performance function as the underlying "ground-truth" reward function'?
	- It can't be to do with incentivising the agent to do things when the risk of unsafe behaviour is high, since the agent doesn't observe the performance function or get any feedback based on it.
- Note how safety information (such as whether the agent has self-modified, Manhattan distance to the water) is provided. More generally, we'll have to find ways for the agent to learn these (or find a way of providing the info to the agent for all/most cases).

## Research directions

- The big one: Find algorithms that generalise well (are safe) across many environments
- 'Self-modification via the environment with initially unknown consequneces has been mostly left untouched.'
- Specifying environments for safety problems such as interpretability, multi-agent problems, formal verification, scalable oversight and reward-learning problems
- Finding generalisations of performance functions (vs fns that are tailored to the specific environment and that don't necessarily generalise to other instances of the same problem)
- 'Can we design off-policy algorithms that are robust to (limited) self-modifications like on-policy algorithms are? More generall, how can we devise formal models for agents that can self-modify?'
- Claimed promising approaches to distributional shift for deep RL (P10):
	- Adapting methods from feedback and control literature to RL case
	- Use of entropy-regularised control laws which are known to be risk-sensitive
	- Incorporation of better uncertainty estimates in neural networks
- Work on safe exploration in combination with deep RL (P11)
- Any agent is incentivised to overcome robustness problems (but they are not incentivised to overcome specification problems), so specification should be a higher priority for safety research
- More research on reward learning, specifically to extend current techniques to larger and more diverse problems and to make current techniques more sample-efficient
- Interpretability and formal verification for deep RL

## Related papers

## Glossary

#### Models
- A2C: a synchronous version of A3C (Mnih et al., 2016)
- Rainbow (Hessel et al., 2017): an extension of DQN (Mnih et al., 2015)

<!-- ## Minor questions
- P3: Why no discounting in env? Because it's simpler?
- P10: Why choose to train each agent on each room separately?
- P11: (Lookup) how is the constraint given to the agent? (Probably as part of the state)
- P14 'For this result (Rainbow not caring about the interruptions) it is important that Rainbow updates on the *actual* action taken (up) when its actions get overwritten by the interruption mechanism, not the action that is proposed by the action (left). This required a small change to our implementation.' -> why? What is the implementation of Rainbow?

#### Knowledge questions
- P3: Why is it a transition *kernel*?
- P4: Off-policy algorithms such as Q-learning are safely interruptible because they are indifferent to the behaviour policy -> why?
	- And the other lit on safe interruptibility...
- P5: Follow up on inaction baseline (lit on avoiding side effects)
- P7: What does it mean by 'the reward function is not potential shaped'?
- P9: 'Classical RL algorithms maximise return in a manner that is insensitive to risk, resulting in optimal policies that may be brittle even under slight perturbations of the environmental parameters.' -> = ?
- P11: What are multi-armed bandits?
- P12: 'A2C tends to be unstable due to occasional entropy collapse.' -> ? (add to A2C file)
	- and A2C particularly sensitive to policy entropy parameter
- P15: What is a Sarsa update rule?
 -->