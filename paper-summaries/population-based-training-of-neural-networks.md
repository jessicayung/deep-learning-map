# Population-based Training of Neural Networks

Jaderberg et. al., Nov 2017

[[arxiv]]](https://arxiv.org/abs/1711.09846) [[DeepMind's blog post]](https://deepmind.com/blog/population-based-training-neural-networks/)

## Summary

Shows how to automate the hyperparameter search process.

## Method: 'Population Based Training' (PBT)

A hybrid of random search and hand-tuning
- Random search: training many neural nets in parallel with random hyperparameters
- Uses information from rest of population to refine hyperparameters
	- Inspired by genetic algorithms
- Periodic exploiting and exploring

![](images/pbt-diagram.png)
*Image credits: DeepMind*

TODO: describe method in more detail.

## Results

- Score: beyond state-of-the-art baselines (claim)
- Compute / time: no computational/time overhead (claim)
- Integrability: 'easy to integrate into existing machine learning pipelines.' (claim)
