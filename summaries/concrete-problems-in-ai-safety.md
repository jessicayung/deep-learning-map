# Concrete Problems in AI Safety

Amodei and Olah et. al., Jul 2016

[[arxiv]](https://arxiv.org/abs/1606.06565) [[Open AI blog post]](https://blog.openai.com/concrete-ai-safety-problems/)

**Tags**: 
- AI Safety
- Reinforcement Learning

## Summary

Discusses five areas (objectives) for AI safety research:
1. [Avoiding negative side effects](#i-avoiding-negative-side-effects)
2. [Avoiding Reward Hacking](#ii-avoiding-reward-hacking)
3. Scalable Oversight
	- TODO: add brief explanation
4. Safe exploration
5. Robustness to distributional shift
	- between train and test

![](images/concrete-problems/problems.png)
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
		- Examples of measures of changes to environment:
			- Naive: state distance
				- BUT agent will try to prevent 'natural' evolution of the environment
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
			- BUT empowerment measures precisions of control over the environment vs total impact.
			- **Research direction (RD)**: Explore variants of empowerment  (TODO: fill in)
3. Multi-agent approaches
	- Understanding other agents and make sure actions don't harm their interests
		- Since avoiding negative side effects is a proxy for avoiding negative externalities
	- Approaches
		- Cooperative Inverse Reinforcement Learning
		- Reward autoencoders
4. Reward uncertainty

Potential experiments:
1. Avoid obstacles (e.g. don't break vases) while accomplishing a single goal.

### II. Avoiding Reward Hacking

Examples of ways reward hacking can happen:
- Closing eyes (don't sense mess) vs cleaning mess up
- Genetic algorithms

#### Ways problems can occur
1. Partially observed goals
	- Agent can only confirm subset of external world through imperfect perception
	- -> Design rewards that represent partial/imperfect measures
	- -> Can be hacked.
		- Though in theory can show there exist rewards as a fn of observations and actions equivalent to optimising true objective function via reducing POMDP to belief state MDP.
2. Complicated Systems
	- Probability that there exists a viable hack increases with agent complexity
		- // bugs in software as complexity of codebase increases
3. Abstract Rewards
	- Sophisticatd reward functions use abstract concepts
	- These may have to be learned by models like neural nets
	- which may be vulnerable to adverse counterexamples <!-- Ok, so? -->
4. Correlation between objective function and accomplishing tasks broken
	- Goodhart's Law
		- 'When a metric is used as a target, it ceases to be a good metric.'
		- Correlation between obj fn and accomplishing task breaks down when obj fn is highly optimised.
			- e.g. using amount of bleach used as a proxy for amount of cleaning done.
	- Feedback Loop
		- Correlation breaks because obj fn has a self-amplifying component.
			- May magnify transient success to perpetual success
5. Environmental Embedding (Hardware)
	- Tampering with reward implementation
	- Often called 'wireheading'
	- Especially concerning if humans are in the loop (may harm or coerce humans to get reward)

#### Approaches
1. Adversarial Reward Function
	- Perspective: Reward hacking as ML system having an adversarial relationship with reward function
	- Solution: make reward function its own agent -> find scenarios where ML system claims is high reward but human labels as low reward
	- Principle: System has multiple pieces trained using different objectives that are used to check each other.
2. Model Lookahead
	- Give reward based on anticipated future states rather than the present state.
	- Helpful for situations where model rewrites reward function
		- by giving negative reward for planning to replace the reward function (lol)
3. Adversarial Blinding
	- Blind a model to certain variables using adversarial techniques
	- e.g. to prevent a model from understanding how its reward is generated.
	- Described as 'Cross-validation for agents'
4. Careful Engineering
	- e.g. formal verification or practical testing of parts of the system
		- can avoid e.g. buffer overflow (TODO: which will cause what?)
	- RD: May be possible to create a highly reliable 'core agent' which could ensure reasonable behaviour from the rest of the agent.
		- *T: like null or 'known safe' policy?*
5. Reward Capping
	- Cap maximum possible reward 
	- To prevent low-probability, high-payoff strategies
	- *T: okay...?*
6. Counterexample Resistance
	- Adversial training, weight uncertainty or architectural decisions to resist adversarial counterexamples
7. Multiple Rewards
	- Use combination of multiple rewards, e.g. different physical implementations or different proxies for the sam informal objective.
		- Difficulty: bad behaviours that affect all reward fns in a correlated way.
	- *T: like repeated codes. RD: parallels in information theory?*
8. Reward Pretraining
	- Defense against agent manipulating its own reward function (feedback or environmental embedding)
	- Train a fixed reward function ahead of time as a supervised learning process divorced from interaction with the environment
	- Difficulty: Forfeits ability to further learn reward fn after pretraining, which may create other vulnerabilities
	- *T: pretrained version of DRLHP?*
9. Variable Indifference
	- Want agent NOT to optimse certain variables.
		- like routing optimisation pressure around parts of the environment.
	- Difficulty: Make sure the variables targeted for indifference are actually the variables we care about, vs proxies or aliased versions.
	- *T: I like this approach.*
10. Trip Wires
	- Deliberately introduce plausible vulnerabilities that will alert us if the agent hacks its reward function.
	- Difficulties: Agent potentially seeing through the trip wire and avoiding it

#### Potential experiments:
- Delusion box environment: standard RL agents distort their own perception to apprear to receive higher reward vs optimising objective
- Creating classes of environments where a delusion box is a natural and integrated part of the environment
	- *e.g. AI Safety Gridworlds watering tomatoes x bucket environment.*

### III. Scalable Oversight

#### Problem
- Providing complete evaluations for complex objectives may be impossible or too costly.
- We then need to reply on cheaper approximations, which can exacerbate problems like unwanted side effects and reward hacking
	- Approximations: calls to proxy, or limited calls to true object function and many calls to a proxy

#### Framework: Semi-supervised reinforcement learning
- Agent can only see its reward on a small fraction of the timesteps or episodes (but evaluated based on reward for all episodes).
- Settings:
	- Active learning: agent can request to see the reward on timesteps where it'd be most useful
	- Random setting: Reward visible on random subset of timesteps
	- Intermediates between active learning and random setting
- Key subtask: 
	- identifying proxies which predict the reward, and 
	- learning conditions under which these proxies are valid
- Authors' thoughts:
	- Use of semi-supervised RL with a reliable but sparse true approval metric may incentivise communication and transparency by the agent.
		- since agent wants as much cheap proxy feedback as possible
	- e.g. for a cleaning robot, 'hiding a mess under the rug simply breaks the correspondence between the user's reaction and the real reward signal, and so would be avoided.'
		- *T: Would there be situations where the action to truly fix a problem is so costly that the agent still chooses to hide a mess under the rug?*

#### Approaches to semi-supervised RL
- Supervised Reward Learning
	- Train model to predict reward from state on per-timestep/episode basis.
		- Use predicted reward to estimate payoff of unlabelled episodes
	- Use weighting or uncertainty estimate to account for lower confidence in estimated vs known reward
- Semi-supervised or Active Reward Learning
	- Supervised Reward Learning combined with semi-supervised or active learning,
	- i.e. Agent requesting to see the reward associated with e.g. 'salient' events in the environment cthat it learns to identify
- Unsupervised Value Iteration
	- 'Use the observed transitions of the unlabeled episodes to make more accurate Bellman updates.' <!-- TODO: ? -->
- Unsupervised Model Learning
	- 'If using model-based RL, use the observed transitions of the unlabeled episodes to improve the quality of the model.' <!-- TODO: ? -->

#### Approaches to scalable oversight other than semi-supervised RL
- Distant supervision
	- Provide useful information about the system's decisions in the aggregate, or
	- Or provide noise hints about correct evaluations
	- e.g.
		- Generalised expectation criteria: ask the user to provide population-level statistics
		- DeepDive: 'asks users to supply rules that each generate many weak labels and extrapolates more general patterns from an initial set of low-recall labeling rules.' 
	- Approach received recent attention in natural language processing
	- RD: Extending and applying this work to case of agents where feedback is more interactive and i.i.d. assumptions may be violated
- Hierarchical reinforcement learning
	- In HRL, top-level agents receive rewards (usually over long timescales) from the environment. To maximise their reward from the enviroment, they act by assigning synthetic rewards for sub-agents, which then assign synthetic rewards to sub-sub-agents. At the lowest level, agents directly take primitive actions in the environment.
	- Relevant because: Sub-agents receive dense reward even if top-level reward is very sparse, since sub-agents are maximising synthetic reward signals defined by higher-level agents.
	- RD: potential proimse of combining ideas from HRL with neural network function approximators <!-- Look into? -->
	- Note: subagents may take actions that don't serve top-level agents' real goals, in the same way that a human may be concerned that the top-level agent's actions don't serve the human's real goals. -> there may be fruitful parallel s between hierarchical RL and several aspects of the safety problem.

#### Potential experiments
	- Semi-supervised RL in basic control environments, such as cartpole balance or pendulum swing-up, 
	- Then basic Atari games
		- Active learning case: is it possible to infer the reward structure from just a few carefully requested samples (e.g. frames where enemy ships are blowing up in Space Invaders?), and thus learn
		- *T: see [Deep Reinforcement Learning from Human Preferences](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/)*
	- Then tasks with more complicated reward structure (simulated or real-world), e.g. robot locomotion or industrial control tasks.

### IV. Safe exploration

- Exploration: taking actions that don't seem ideal given current information, but which help the agent learn about its environment.
- Problem: Can be dangerous because it involves taking actions whose consequences the agent doesn't know well
	- **Can we predict which actions are dangerous and explore in a way that avoids those actions?**
- Much literature on this topic

#### Approaches
- Hard-coding avoidance of catastrophic behaviours: is not generalisable
	- Works well when there are only a few things that could go wrong, and the designers know all of them ahead of time.
- Risk-Sensitive Performance Criteria
	- Changing the optimisation criteria from expected total reward to other objectives that are better at preventing rare catastrophic events
	- Approaches:
		- optimising worst-case performance
		- ensuring that the probability of very bad performance is msmall, or 
		- penalising variance in performance
	- Status:
		- Methods not yet tested with DNNs, but should be possible in principle for some methods
	- Related lines of work:
		- Estimating uncertainty in value functions that are represented by deep neural networks
		- Using off-policy estimation to perform a policy update that is good with high probability
- Use Demonstrations
	- Avoiding the need for exploration if we use inverse RL or apprenticeship learning, where the learning algorithm is provided with expert trajectories of near-optimal behaviour.
	- Ways of using demonstrations
		- Train only on demonstrations
		- To reduce the need for exploration in advanced RL (by training on a small set of demonstrations)
		- To create a baseline policy, such that even if further learning is necossary, exploration away from the baseline policy can be limited in magnitude.
- Simulated Exploration
	- Do exploration in simulated environments instead of the real world
	<!-- - Note: 'in systems that involve a continual cycle of learnnig and deployment, there may be interesting research problems associated with how to safely incrementally update policies given simulation-based trajectories that imperfectly represent the consequences of those policies as well as reliably accurate off-policy trajectorios (e.g. semi-on-policy evaluation).' -->
- Bounded Exploration
	- Remain within portion of state space we know to be safe
		- Some definitions of safety:
			- Remaining with an ergodic region of the state space s.t. actions are reversible
			- Limiting P(X) of huge negative reward to some small value
			- OR obey constarints on separate safety function with high probability (safety functions, performance functions separate)
		- If model exists, can use model to extrapolate forward and ask whether an action will take us outside the safe state stpace
	<!-- TODO: add -->
	- Potentially related areas: H-infinity control, regional verification
- Trusted Policy Oversight
	- Limit exploration to actions the trusted policy believes we can recover from.
		- Requires a trusted policy (policy we trust to be safe) and a model of the environment
- Human Oversight
	- Check potentially unsafe actions with a human.
	- But runs into the scalable oversight problem: there may be too many actions, or they may occur too quickly for a human to judge them
	- Challenges: 
		- Having the agent be a good judge of which actions are genuinely risky
		- Finding appropriately safe actions to take while waiting for oversight

#### Potential experiments
- Suite of environments where agents can fall prey to harmful exploration, but there is enough pattern to catastrophes that clever agents can predict and avoid them.

### V. Robustness to Distributional Change
- Problem: In unfamiliar situations, ML systems are often highly confident in their erroneous classifications.
	- Need to learn to recognise their own ignorance
- Typical constraints
	- Likely have access to a large amount of labelled data at training time, but little or no labelled data at test time
- Goal
	- Ensure that model 'performs reasonably' on potetentially different test dist p\*, i.e.
		- often performs well on p\*,
		- knows when it is performing badly (and ideally can avoid/mitigate the bad performance by taking conservative actions or soliciting human input).
- Potentially relevant areas:
	- Change detection and anomaly detection, hypothesis testing, transfer learning

#### Approaches

#### A. Detecting  when a model is unlikely to make good predictions on a new distribution, or trying to improve the quality of those predictions

- Well-specified models: 
	- e.g. covariate shift and marginal likelihood
	- Approach (1): Make covariate shift assumption 
		- Assumption: that p_0(y|x) = p^\*(y|x).
			- where x is input, y is output, p^\* is test dist, p_0 train dist.
			- A strong and untestable assumption
		- Method (1): Sample re-weighting: re-weight each training example by p^\*(x)/p_0(x).
			- -> then can estimate performance on p^\* or even re-train model to perform well on p^\*.
			- Limited by variance of importance 	estimate, which is large or even infinite unless p_0 and p^\* are close together
		- Method (2): Assume well-specified model family
			<!-- TODO: could add more, but bit technical, dk if it'll help -->
	- Approach (2): Build generative model of the distribution
		- Free to assume other invariants (e.g. p(y) changes while p(x|y) stays the same) instead of assuming that p(x) changes while p(y|x) stays the same.
		<!-- Todo: what is p exactly -->
			- Advantage: such assumptions usually more testable since they don't only involve unobserved var y
			- Disadvantage: generative approaches are even more fragile than discriminative approaches in the presence of model-misspefication
	- Approaches above all rely strongly on having a well-specified model family
		- Possible to mitigate this with very expressive models such as kernels, Turing machines or very large neural networks
			- But may not be able to learn specification of model given finite data 
- Partially-specified models: 
	- e.g. method of moments (econometrics), unsupervised risk estimation, causal identification, and limited-information maximal likelihood (even instrumental variables)
	- Assumes constructing a fully well-specified model family is infeasible
	- Partially specified models: models for which assumptions are made about some aspects of a distribution, but for which we are agnostic or make limited assumptions about other aspects.
		- e.g. linear regression with white noise: can identify W without knowing precise dist of noise.
	- Current ML:
		- Success for method of moments in estimation of latent variable models
<!--			- Focus is on using moM to overcome non-convexity issues, but it can also offer a way to perform unsupervised learning while relying only on conditional independence assumptions (vs strong distributional assumptions underlying maximum likelihood learning) -->
		- Some work focusing only on modelling the distribution of errors of a model
			- Goal: perform unsupervised risk estimation (given a model and unlabeled data from a test distribution, estimate the labeled risk of a model)
				- Can potentially handle very large changes between train and test
			- *T: nice, look into this problem*
	- RD / Q: Are partially specified models fundamentally constrained to simple situations and/or conservative predictions?
- Training on multiple distributions
	- Train on multiple training distributions in the hope that a model which simultaneously works well on many training dists will also work well on a novel test dist
	- Still important to detect when one is in a situation that is not covered by the training data and to respond appropriately
	- Can combine with approaches above
	- Lit:
		- Applies in automated speech recognition systems

#### How to respond when out-of-distribution
- Ask humans for information 
	- But may be unclear what questions to ask
		- -> work on pinpointing aspects of a structure that a model is uncertain about
		- obtaining calibration in structured output settings <!-- TODO:? -->
	- May not be an option in time-critical situations
		- Relevant work based on reachability analysis or robust policy improvement
		- RD: Combine this work with methods for detecting out-of-dist failures in a model
- Non-structured output setting (agents that can act in an environment, e.g. RL agents)
	- Inormation about the reliability of percepts (observations?) in uncertain situations seems to have great potential value
		- Potential agent responses if info seems uncertain
			- Gather info that clarifies percept (e.g. move closer to speaker in a noisy environment)
			- Engage in low-stakes experimentation when uncertainty is high (e.g. try potentially dangerous chemical reaction in a controlled environment)
			- Seek experiences that are likely to help expose the perception sytem to the relevant distribution (e.g. practice listening to accented speech)
- Generally a RD

#### Frameworks
- A unifying view: counterfactual reasoning and ML with contracts
- Counterfactual reasoning
	- 'What would have happened if the world were different in a certain way?'
	- Thinking of distributional shifts as counterfactuals
- Machine learning with contracts
	- Construct ML systems that satisfy a well-defined contract on their behaviour, analogous to software systems
	- Brittle implicit contract in most ML systems: that they only necessarily perform well if the training and test distributions are identical.
		- Condition difficult to check and rare in practice, valuable to build systems that perform well under weaker contracts that are easier to reason about
	- Approaches: 
		- Partially specified models (above)
		- Reachability analysis
			- optimise performance subject to the condition that a safe region can always be reached by a known conservative policy
		- Model repair
			- Alter a trained model to ensure that certain desired safety properties hold

#### Potential Experiments
- Challenge: Train a state-of-the-art speech system on a standard dataset that gives well-calibrated results on a range of other test sets, like noisy and accented speech
- Design models that could consistently estimate (bounds on) their performance on novel test distributions
- Create an environment where an RL agent must learn to interpret speech as part of some larger task, and to explore how to respond appropriately to its own estimates of its transcription error
	- TODO: why is this valuable?
<!--
 ## Thoughts

## Related papers

-->
  
### Related Efforts
<!-- TODO: add list of related research efforts with links-->
- Related Problems in Safety
	- Privacy 
	- Fairness
	- Security (Adversarial atttacks against a legitimate ML system)
	- Abuse (Misuse of ML systems to attack or harm people)
	- Transparency (Understading what complicated ML systems are doing)
	- Policy