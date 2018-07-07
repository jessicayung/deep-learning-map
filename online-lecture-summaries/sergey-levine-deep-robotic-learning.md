# Deep Robotic Learning (Sergey Levine)

[Talk](http://people.eecs.berkeley.edu/~svlevine/)

Why is there still a significant gap between robots and humans (fluid action), e.g. for the door opening task?

## End-to-end vs modular pipeline approach
- Writing software: usually use separate abstractions (modules), take a pipeline approach
	- Problem: Leaky abstractions: when you make a mistake, mistakes accumulate. So you need to slow things down
- End-to-end vs Pipeline approach: interface between layers is learned vs hand-designed

### Additional complexity with Robotics (vs other Deep Learning tasks)
1. don't have direct supervision in sensorimotor loop: 
	- don't have labels of correct motor commands you need to apply for each sensory input, need to figure those out on its own
2. Actions have consequences: 
	- have dims of time, have causality
	- may have to perform actions that are not immediately useful but have consequences down the line

Reinforcement Learning is one way of dealing with these two problems.

### Shortcuts as a different kind of abstraction
- e.g. motor skill of catching a ball
	- complicated pipeline: look at ball, estimate velocity, roughly plan where it's going ta land, plot plan and see where to catch it
	- vs gaze heuristic: look at it and start running, run faster if it drops vs slower if it rises in gaze.
	- reasoning about task heuristically vs complicated pipeline -> can find shortcuts
	- CLAIM: these shortcuts are abstractions that may be building blocks of higher-level understanding

### End-to-end training can optimise for the correct objective when modular approaches fail
- e.g. exp: combining perception and control vs separating them: is this better for a certain task?
	- task: put shape into shape-sorting cube
	- CV for vision module
	- result: surprisingly separated perception + control modules achieved 0% success, vs 96.3% for end-to-end (latter not as surprising bc simple task)
		- why separate modules didn't work: 
			- vision module 1.5cm accuracy (pretty good), but error allowance for task is a few mm.
		- why end-to-end worked:
			- optimised for right objective: only need to estimate two dimensions, find right kind of mistakes you can make


## On Deep RL (Advs and disadvs)
- Advs
	- General-purpose algorithms
		- e.g. combine vision and control and get improved performance
	- ...

- Disadvs
	- High sample complexity (need lots of experience)
		- e.g. pong 2-10M frames
	- Emphasis on mastery, not generalisation
		- like testing on a training set
	- Also CV results on pre-training, initialisation and fine-tuning don't seem to carry over


## Work in the Robotic AI and Learning Lab
1. Can we develop more sample-efficient deep RL algorithms?
	- Mirror descent guided policy search
	- Mixing model-based and model-free updates
	- Parallelising MDGPS
2. Studying Generalisation with deep RL
	- Collecting varied real-world data
	- Learning predictive models from vision
	- Setting goals
	- Learning easily adaptable features to accelerate future learning

1. policy search
- dist of actions u_t over states x_t.
- note in reality searching over policies, not parameters: NN-parameterised policies happen to lie on a manifold
	- may be worth it to go off the manifold and then jump back on in special cases (e.g. multimodal)
- From this idea: Mirror Descent Guided Policy Search
	- Mirror descent: constarined opt alg
		- goal: $$\min_x f(x)$$ s.t. $$x\in X$$.
		- 1. $$\mathbf{x}^{k+\frac{1}{2}} \leftarrow \min_x \hat{f}(x)$$ s.t.$$D(x,x^k)\leq \epsilon$$
			- find minimum of f in constraint-ball around initial x0 (trust region)
		- 2. $$x^{k+1} \leftarrow \min_{x}D(x,x^{k+\frac{1}{2}})$$ s.t. $$x\in X$$
			- project point back to constraint manifold
		- D some arbitrary divergence
			- e.g. Euclidean dist: projected gradient descent
	- applied to policy search: $$\pi$$ as $$x$$, use KL divergence since $$\pi$$ is a probability distribution
		- improvement step tricky
		- projection step simpler, can do by supervised learning
	- How to improve the policy
		- approximate policy with other policies, e.g. time-varying linear Gaussion policies $$p_i(u_t|x_t) = N(K_{t,i}x_t+k_{t,i}\sum_{t,i})$$.
			- can't represent bifurcation though
			- simple ways of improving policy
		- 1 model-based approach: locally approx dynamics (locally linearise dynamics, use local LQR-like gradient)
			- do for each component separately
			- works well in smooth, continuous dynamics regions
		- 2 model-free approach: path integral policy alg
			- take samples from initial state, refit time-varying linear Gaussian so good samples have higher probability
			- pick actions that get better reward: much slower but works well near discontinuities that can't be modelled well
		- 3 mix model-based and model-free approaches
			- model-free on residual: make traj more likely if (reward - predicted reward) for trajectory is high

- Sample Efficiency
	- Compare in reacher (simulated task)
		- DDPG is like an actor-critic algorithm
		- 10x to 100x difference (vs DDPG, TRPO)
	- real-world task for 40mins 
		- vs 300min deep Q-learning (model-free, comparable to continuous DDPG)

- Improvement step can be parallelised
	- send improved actions to replay memory, separate workers sample from pooled replay memory and improve policy
	- generalise to different door handles and orientations

## Generalisation with deep RL

Important ingredients for generalisation in supervised learning
- Computation (large amount of)
- Algorithms (improvement in)
- Availability of large amount of data (Sergey thinks this is v important)

Robotics:
- Algos and models not as far along
- Data: :(
	- need to be able to train on varied experience
	- robots collab to learn single sensorimotor skill, here grasping (useful and )

- Grasping by hand-eye coordination 
	- monocular camera (no depth) so cannot just memorise position
	- no prior knowledge except finger-closing heuristic to see if they've been successful
	- model: learning a critic
	- results:
		- learns material properties before geometry (e.g. for soft objects pinches it in the middle, vs hard ones at sides)
		- repositions a lot with small, flat objects
	- Data: 800,000 grasps collected over several months over wide variety of training objects

- Q: how well does this transfer across robots?
	- i.e. if use both kooka arm and custom arms, can you get higher success rate vs just data from the same robot? (reasonably similar physics since fingers the same)
	- result: substantial improvement, esp when you have v little data (e.g. 100k images)

### proj 2
- Q: Can we learn more general models useful for a variety of tasks? 
	- data collected more diverse: push objects around and recording resulting interactions, try to use interactions to learn about physics
	- predict videos from an original frame
		- want to use predictions to take useful actions in the real world
		- user specifies goal (location of object), robot tries to pick actions where model predicts object ends up in user-specified location
			- robot does not know about objects, only image pixels and actions
		- result: pretty good
			- continuous control: re-plan every 250ms

- Q: want to set more complex goals

### project
- robot learns model of rope through interactions
- set subgoals and goal, robot tries to reach goals by hitting subgoals (images)
- reward or goal/subgoals learned based on images

### project
Learning easily adaptable features to accelerate future learning

- when train big CNN on ImageNet, can take features for new visual perception tasks
	- features likely capture useful things
	- Q: what are imagenet features of motor control or robotic learning?
- trouble with RL benchmarks
	- small-scale vs large-scale
	- emphasis on mastery vs diversity
	- evaluated on performance vs generalisation

- different tasks
	- MAML: update parameters such that you can improve as much as possible when updating based on gradient descent
		- learn neutral parameter value that can be quickly adapted to new tasks
		- result: learns forward-backward in one iteration from learned 'neutral parameters'


## q&a
- updating policy more frequently speeds up training
- using synthetic data: don't need super realisitc data to generalise to real world, if data is very diverse, can generalise (real wordl in convex hull of dataset)
- nice to have ImageNet of robotics (simulated or real world), and to have more complex tasks
- what does it mean to have a feature in robotics or motor control
	- may need to be sth that couples perception and control
- how to have hierarchical models that capture the salient features of the world
	- hypotheses:
		- maybe we haven't scaled up enough, and when we have we might see those things emerge (using naive methods)
		- fancier: models that try to explicitly reason about properties of objects
		- mult forms of supervision
- how to get to 0% error in grasping task
	- grasping objects here are difficult
	- some mistakes that suggest more sophisticated needed: e.g. manipulation prior to the grasp (need better RL to extract strategies at this scale)
- how find most informative tasks for grasping
	- here no attempt to diversify training set (robots just pick up what they want to), using proper exploration methods to diversify training set should go a long way
- 