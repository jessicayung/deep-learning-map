# Population-based Training of Neural Networks

Jaderberg et. al., Nov 2017
[arxiv](https://arxiv.org/abs/1711.09846) [DeepMind's blog post](https://deepmind.com/blog/population-based-training-neural-networks/)

## Summary

Shows how to automate the hyperparameter search process.

## Method: 'Population Based Training' (PBT)

A hybrid of random search and hand-tuning
- Random search: training many neural nets in parallel with random hyperparameters
- Uses information from rest of population to refine hyperparameters
	- Inspired by genetic algorithms
- Periodic exploiting and exploring

![](images/population-based-training-of-neural-networks.png)

TODO: describe method in more detail.

#### Method in more detail
'By combining multiple steps of gradient descent followed by weight copying by exploit, and perturbation of hyperparameters by explore, we obtain learning algorithms which benefit from not only local optimisation by gradient descent, but also periodic model selection, and hyperparameter refinement from a process that is more similar to genetic algorithms, creating a two-timescale learning system.'

## Results

- Score: beyond state-of-the-art baselines (claim)
- Compute / time: no computational/time overhead (claim)
- Integrability: 'easy to integrate into existing machine learning pipelines.' (claim)
- 