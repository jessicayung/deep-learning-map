# Interactions Between Model-Free and Model-Based Reinforcement Learning

Peter Dayan, Oct 2011 

Seminar Series from the Machine Learning Research Group at the University of Sheffield

## Three decision-makers (controllers)

1. Tree search
	- Problem is usually that search space (breadth and depth) are too large
	- Neuroscience: 
		- Tolmanian forward model: belief that animals would build model of task to solve and search model
			- forward/backward search, unsure then
			- Experiment: 
				- Procedure
					- Train hungry mouse to press lever to get cheese (short or extensive training)
					- Devalue cheese: pair no-lever cheese with sickness
					- Later, test whether animal is willing to press the lever
				- Results
					- If short training, much less willing to press lever after cheese devalued
						- Model-based, cognitive mapping?
					- If extensive training, only near-negligibly less willing to press lever
			- Related brain areas (from other experiments): OFC, dIPFC, dorsomedial striatum, BLA -> building evaluations for searcing process
		- Characteristics
			- Statistically efficient: 'every piece of knowledge you can learn in just the right time', learning simple 
				- no temporal complexity, instant info / feedback on what happens
			- computationally catastrophic
				- to use that knowledge, bc you have to search complex tree for downstream consequences
2. Position evaluation (Habit/Model-free)
	- e.g. how likely am I to win the game of chess if I make this move?
	- If perfect, don't need to build tree
	- RL: learning values without building trees
	- Minimise inconsistency (error) between successive predictions
		- e.g. if your evaluator says A is a good move (and you make that move), you should not be in a significantly worse position after your opponent makes a move
		- trying to predict the future: the sum of future rewards $V(x(t))=r(t)+r(t+1)+...=V(x(t+1))$ (RHS if predictions are correct)
		- Temporal difference error (TD error) $\delta(t) = r(t)+V(x(t+1))-V(x(t))$
			- difference between LHS and RHS of prev eqn
		- Model-Free
			- in the sense that you don't have to build a model of the world
			- cached in the sense of saving result of past experience
	- s
3. Situation memory
	- 