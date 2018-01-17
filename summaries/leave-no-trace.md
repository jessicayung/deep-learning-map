# Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning

Eysenbach et. al., Nov 2017

Authors: Benjamin Eysenbach, Shixiang Gu, Julian Ibarz, Sergey Levine

[[arxiv]](https://arxiv.org/abs/1711.06782) [[Videos]](https://sites.google.com/site/mlleavenotrace/)

**Tags**: 
- AI safety: avoiding side effects, reversibility
- ICLR 2018

## Summary

Presents an autonomous method for safer reinforcement learning: 
- By learning a reset policy as well as a forward policy, we can determine when the forward policy is about to enter a non-reversible (unsafe) state. 
- We can thus abort the process before entering these states, allowing for safer RL (especially exploration).


## Method

Simultaneously learns a forward and a reset policy.
- Reset policy resets the environment for a subsequent atttempt.
	- Learn a value function for the reset policy -> can automatically determine when forward policy is about to enter a non-reversible state
- Can have uncertainty-aware safety aborts

## Results

<!--TODO: what is a manual reset? ok i think i get it, might expand on this after reading paper-->
- Reduced number of manual resets required to learn a task
- Reduced number of unsafe actions that lead to non-reversible state
<!--TODO: the curriculum stuff -->
- 'Can automatically induce a curriculum.'

## Related papers
- [Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](https://arxiv.org/abs/1703.05407)
	- Recommended by Jack Clark (ImportAI)