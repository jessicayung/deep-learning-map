# L1 Intro

Lectured by Sergey Levine, Fall 2017. ([Course website](http://rail.eecs.berkeley.edu/deeprlcourse/))

## Reinforcement Learning
- Loop of decisions/actions and consequences (observations, rewards) between agent and environment
- Why care about deep RL?
	- DL: end-to-end training of expressive, multi-layer models, typically NNs
	- deep models are what allow RL algorithms to solve complex problems end-to-end.
		- models that can represent solutions to those complex problems
- What does end-to-end learning mean for sequential decision making?
	- Combine modules such as vision NN -> net -> output action into a single sensorimotor loop
		- e.g. 
		- each stage designed manually, pull out abstractions -> loses information. need to make sure abstractions are the right ones. But we don't know what abstractions are correct.
	- Q: what does it mean for learning to be end-to-end vs modular in a pipeline? If the whole model is differentiable (including feeding in variables from one part to another) but you have separate networks for vision and action, is that still end-to-end?
		- Layers of representation adapted to task, don't need to manually craft (1) abstractions and (2) interfaces, these are optimised with end-to-end training. (So above example seems to still be end-to-end.)
- any supervised learning problem can be reformulated as an RL problem
	- though it might not always be a good idea
		- since RL makes fewer assumptions than SL
	- esp useful when metrics are not differentiable, e.g. BLEU score for translation


## key challenges of deep RL
- no direct supervision
	- indirect: did it get reward vs was the action correct
	- know what you want but not how you get it
- actions have consequences: need to make entire sequence of 'correct' decisions vs just one, s.t. you get the reward

## why now?
- Recent advances in DL 
- Recent advances in RL (scalable algs)
- Recent advances in computational capability: practical to train models in feasible timescales
- but note core ideas have been around for a long time

### Examples of recent results
- Atari games (Q-learning or policy gradients), 
- real-world robots (guided policy search or q-learning), 
- Go (supervised learning + policy gradients + value functions + MCTS)

## other problems we need to solve to enable real-world sequential decision-making

(other than RL and DL)

- Learning without a reward
	- Inverse RL: learning reward functions from examples
	- Transferring skills between domains
	- Learning to predict and using prediction to act
		- understand physics but not know reward
- Problem of reward:
	- e.g. even evaluating reward fn for pouring glass of water is v difficult, CV problem + unnatural notion of reward
	- 'as humans, we are accustomed to operating with...sparse' rewards
	- basal ganglia: associated with reward
- Potential solutions: other forms of supervision
	- Learning from demonstrations
		- directly copying observed behaviour (imitation learning)
		- inferring intent: inferring rewards from observed behaviour (inverse RL)
	- Learning from observing the world
		- learning to predict (understanding physics of environment)
			- map  from (s,a) to (snext, anext)
			- neuro: we predict the consequences of our motor commands. and also in the visual cortex.
				- use mental simulations instead of relying only on trial and error
				- if know model:
					- can do complicated stuff IF you have a perfect model
				- or don't know model, can learn model
					- e.g. robot arm takes random actions and record actions and images (consequences) -> learns to predict images (Finn et. al., 2017, recurrent CNNs)
						- given image / point goal, take action(s) that model predicts will get to goal
						- (model-based)
		- unsupervised learning
			- picking up structural regularities in the world -> accelerate learning
	- Learning from other tasks
		- transfer learning
		- meta learning: using past experience of learning other tasks to figure out how to learn more quickly 
			- subtly different from transfer learning
			- e.g. don't know how to do X, but prev solved tasks using A,B,C, so I'll try those.

## how can we build an intelligent machine
- components and abstractions?
- if build component-by-component mimicking the brain, each component will likely be brittle, and not be able to adapt as well
- so consider learning as the basis of intelligence
	- some things we can do 'naturally' like walking
	- vs driving cars: have to learn
	- humans can learn very complex things, so learning mechanisms likely powerful enough to do everything we associate with intelligence
		- may still hard-code a few important bits
- so build learning algorithms for each module, or use a single flexible algorithm?
	- if for each module: may be harder? because multiple and interfaces?
	- evidence that perception-wise may be a single one: can build electrode array to camera and can learn to perceive world with eyes closed using your tongue.
		- alse 'ferret rewiring experiment': optic nerve reconnected to auditory cortex, have visual acuity after a period: can learn to 'see' from signals coming from auditory nerve.
- requirements of the 'single algorithm'
	- interpret rich sensory inputs
	- choose complex actions
- why might deep RL satisfy these requirements?
	- deep = can process complex sensory input
	- RL -> can choose complex actions
- deep RL and brain
	- 'Unsupervised learning models of primary cortical receptive fields and receptive field plasticity' (Saxe et. al., 2011)
		- vision, sound, touch, fMRI. statistics of features when exposed to diff stimuli.
			- doesn't mean NN works like the brain does - may be in the environment and any complex-enough system can extract those features. is fine.
			- tactile: from somatosensory cortex of monkey touching drum with indentations, vs DL white dust deposited on glove, can see pattern where person touched the object or sth.
- RL and brain
	- TD and dopamine
	- basal ganglia appears to be related to reward system
	- model-free RL-like adaptation is often a good fit for experimental data of animal adaptation (but not always)

- what DL and RL can do well now?
	-  acquire high degree of proficiency in domains governed by simple, known rules (e.g. games)
	- learn simple skills with raw sensory inputs, given enough experience (in real world, e.g. grasping)
	- learn from imitating enough human-expert behaviour (e.g. SDCs)
- what DL and RL can't do well yet
	- learning quickly (perhaps by leveraging past experience or figuring out which dims to adapt in)
	- reusing past knowledge (transfer learning in deep RL open problem)
	- unclear what reward function should be
	- unclear what role of prediction should be