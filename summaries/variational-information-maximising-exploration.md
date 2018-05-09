# Variational Information Maximising Exploration

Houthooft et. al., Jan 2017

Authors: Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel

[[arxiv]](https://arxiv.org/abs/1605.09674)

**Tags**: 
- Bayesian Neural Networks
- Reinforcement Learning

## Summary

Method of curiosity-driven exploration (seeking state-action regions that are relatively unexplored): choose action that reduces uncertainty (entropy) in agent's belief about environmental dynamics.

Estimate posterior dynamics distribution (generally intractable) using variational Bayes, i.e. by maximising variational lower bound/free energy.

## Method

$\max\limits{\{a_t\}}\sum_t(H(\Theta|\zeta_t, a_t)-H(\Theta|s_{t+1},\zeta_t, a_t))=\max\limits{\{a_t\}}\sum_t(I(s_{t+1},\Theta|\zeta_t, a_t)$, 

where $\zeta_t=\{s_1,a_1,...,s_t\}$ is the history of the agent up till time t. H(.) is entropy, I(.) is mutual information.

Equivalent to $\max\limits{\{a_t\}}\sum_t(E_{s_{t+1}~P(.|\zeta_t, a_t)}[D_{KL}[p(\theta|\zeta_t, a_t, s_{t+1})||p(\theta|\zeta_t)]]$

i.e. KL between agent's new belief over dynamic model (posterior) and agent's old belief over dynamics model (prior). Interpret this KL as information gain.

Further details: 
- Model agent dynamics using a Bayesian NN parameterised by fully factored Gaussian distribution.

## Results
- explores and performs well: 'significantly better than heuristic exploration methods across various continuous control tasks and algorithms'

## Future work
- Investigate measuring surprise in the value function
- using the learned dynamics model for planning.

## Thoughts
- Follow up on intuition on comparing information gain with compression improvement (intrinsic reward objective)

