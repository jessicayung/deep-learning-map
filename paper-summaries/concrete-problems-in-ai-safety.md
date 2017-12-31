# Concrete Problems in AI Safety

Amodei and Olah et. al., Jul 2016

[[arxiv]](https://arxiv.org/abs/1606.06565) [[Open AI blog post]](https://blog.openai.com/concrete-ai-safety-problems/)

**Tags**: 
- AI Safety
- Reinforcement Learning

## Summary

Discusses five areas (objectives) for AI safety research:
1. Avoid negative side effects
2. Avoid Reward Hacking
3. Scalable Oversight
	- TODO: add brief explanation
4. Safe exploration
5. Robustness to distributional shift
	- between train and test

![](concrete-problems/problems.png)
<!--
- Wrong objective function
- Correct objective function, have method to evaluate it
-->
For each area, they discuss the research problems, current and promising approaches and potential experiments to conduct.


(Work in progress: writing much more detailed summary)

## Detailed summary

### Why we need to address these problems
1. Increasing promise of reinforcement learning (RL), which allows agents to have **highly intertwined interactions with the environment**.
2. Trend towards more complex agents and environments
	- Increases probability of side effects occcuring
	- Agent may need to be sufficiently complex to hack rewards
3. Trend towards increased autonomy of AI systems
	- (in terms of action space in the real world, i.e. directly controlling industrial processes vs outputting recommendations to users)
	- Increases potential harm caused by systems that humans cannot necesarily correct or oversee

### I. Avoiding negative side effects

#### Difficulties:
- Not feasible to identify and penalise every possible disruption

#### Approaches
1. Using impact regularisers
	- impact regularisers: penalise changes to environment.
	1. Define impact regulariser
		- Naive: state distance
		- Distance between future state under agent's policy and future state under a reference policy 
			- Can be v sensitive to (1) representation of state and (2) distance metric.
			- Reference policy examples:
				- 'Null' policy: where agent acts passively 
					- (May be hard to define if starting point is e.g. agent carrying a heavy box, where doing nothing is not a passive action)
				- 'Known safe' policy
	2. Learn impact regulariser
		- Learn via training over many tasks (transfer learning)
			- since side effects may be more similar across tasks than the main goals
			<!-- 
			- Separating side effect components from main tasks may speed up transfer learning.
			- (Similar to model-based RL approaches that attempt to transfer a learned dynamics model but ...) 
			-->
2. Penalise influence
	- Prefer agent not to get into a position where it could easily do things which have side effects.
	- Measures of influence (mostly info theoretical):
		- Empowerment
			- := max possible mutual information between agent's potential future action and its potential future state
				= Shannon capacity of channel between agent's actions and the environment.
			- BUT ...
3. Multi-agent approaches
4. Reward uncertainty

<!-- ## Thoughts

## Related papers

-->