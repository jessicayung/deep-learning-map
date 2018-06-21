# Distributed
- [Distributed Deep Reinforcement Learning: learn how to play Atari games in 21 minutes (Adamski, Adamski, Grel, Jędrych, Kaczmarek and Michalewski, Jan 2018)](https://arxiv.org/abs/1801.02852)
	- Model: Batch A3C (BA3C) with Adam optimiser, large batch size of 2048
	- Scale up algorithm using distributed systems techniques, which allow them to run their algo in 21 minutes across 64 workers comprising 768 distinct CPU cores.
		- vs 10h when using a single node with 24 cores when using a baseline single-node implementation