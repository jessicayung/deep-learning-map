# Deep RL Doesn't Work Yet

1. v sample inefficient 
	- Atari: (e.g. RAINBOW uses c. 18M frames (p good actually) per game to reach median human performance, i.e. 83h play experience + training time)
	- MuJoCo: input state position and velocity of each joint of simulated robot. Benchmarks take 1e6 to 1e8 steps to learn, don't have to solve vision.
	- Sometimes easy to generate experience / samples (like games), and not a problem in those cases

2. If only care about final performance, many problems are better solved by other methods
	- MuJoCo: model-predictive control: planning against ground-truth world model (physics simulator)
	- Atari: MCTS (UCT agent)
	- Price of generality (hard to exploit problem-specific info that could help with learning, forces you to use many examples to learn things that could've been hardcoded)
	- BUT AlphaGo unambiguous win for Deep RL

3. usually requires a reward fn and reward fn design is difficult
	- reward fn must capture exactly what you want bc deep RL tends to overfit to reward
	- exceptions: e.g. inverse RL
	- reward design difficult bc needs to encourage behaviours you want while still being learnable
	- shaped rewards can bias learning, sparse rewards hard to learn

4. often gets stuck in local optima (even when given a good reward)
	- exploration-exploitation is hard
	- e.g. MuJoCo: cheetah running on its back
	- people have been exploring approaches like intrinsic motivation, curiosity-driven exploration, count-based exploration. but probably hard to find single trick that works consistently across all envs.

5. 
	

6. Final results unstable and hard to reproduce
- 25-30%+ of runs for a soln that 'works' can fail just bc of random seeds

### Other comments
- Karpathy comment: 
	- ?? bottlenecked by credit assignment or supervision bitrate, not by a lack of a powerful representation

---
“Deep Reinforcement Learning That Matters” (Henderson et al, AAAI 2018). Among its conclusions are:
	•	Multiplying the reward by a constant can cause significant differences in performance.
	•	Five random seeds (a common reporting metric) may not be enough to argue significant results, since with careful selection you can get non-overlapping confidence intervals.
	•	Different implementations of the same algorithm have different performance on the same task, even when the same hyperparameters are used.

