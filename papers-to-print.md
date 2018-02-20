
# Papers to print

## Next papers to look at
- [Feature Visualisation](https://distill.pub/2017/feature-visualization/)
- [Regularised Evolution for Image Classifier Architecture Search (Real and Aggarwal et. al., Feb 2018)](https://arxiv.org/abs/1802.01548)
	- Comparing regularised evolution, non-regularised evolution and reinforcement learning for image classifiers
	- Regularised evolution sets new state-of-the-art ofr CIFAR-10, mobile-size ImageNet and reaches top 5 accuracy for ImageNet
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
	- Transfer learning, multi-task RL
	- 30 tasks on DeepMindLab environment (DMLab-30) and 57 Atari games
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (Silver et. al., Dec 2017)](https://arxiv.org/abs/1712.01815?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)
- [The Case for Learned Index Structures (Kraska et. al., Dec 2017)](https://arxiv.org/abs/1712.01208)
	- Show how to implement neural-network-based 'learned indexes' that can replace B-Tree indexes.
		- Result: 70% speed improvement, save an order of magnitude in memory over several real-world datasets
			- Caveats: dataset fixed etc
	- To consider: implications of replacing indices with learned models for system design, software etc
	- Related:
		- [Software 2.0: Related article by Andrej Karpathy on how neural networks represent a fundamental shift in how we write sofware (Nov 2017)](https://medium.com/@karpathy/software-2-0-a64152b37c35)
		- [DeepConfig: Automating Data Center Network Topologies Management with Machine Learning (Streiffer and Chen et. al., Dec 2017)](https://arxiv.org/abs/1712.03890)
			- TODO: write summary
- [Efficient Neural Architecture Search via Parameter Sharing (Phan and Guan et. al., Feb 2018)](https://arxiv.org/abs/1802.03268)
	- ENAS: Fast and inexpensive approach for automatic model design. Claim 1000x less computationally expensive than standard Neural Architecture search.
	- Controller trained with policy gradient to select subgraph (to discover network archis).
		- Model corresponding to subgraph trained to min cross-entropy loss.
	- Sharing parameters among child models -> strong empirical performance while using fewer GPU-hours. 
		- 1000x less (computationally?) expensive than standard Neural Architecture search
	- e.g. 
		- state-of-the-art among methods without post-training processing for Penn Treebank: Text Perplexity 55.8 
		- CIFAR-10: 2.89% test error
- [Superhuman AI for heads-up no-limit poker: Libratus beats top professionals (Brown et. al., Dec 2017)](http://science.sciencemag.org/content/early/2017/12/15/science.aao1733)
- [Peephole: Predicting Network Performance Before Training (Deng et. al., Dec 2017)](https://arxiv.org/pdf/1712.03351.pdf)
- [Mathematics of Deep Learning (Vidal et. al., Dec 2017)](https://arxiv.org/abs/1712.04741)
	- 'Review of recent work that aims to provide a mathematical justification for several properties of deep networks, such as global optimality, geometric stability, and invariance of the learned representations.''
- [Symmetric Decomposition of Asymmetric Games](https://www.nature.com/articles/s41598-018-19194-4)
	- Decomposing asymmetric games into its symmetric counterparts (which consider the reward table of each player as a symmetric 2-player game with eqm points that coincide with the original asymmetric game)
- [Deep Learning: A Critical Appraisal (Jan 2018)](https://arxiv.org/abs/1801.00631)
	- Ten concerns for DL
	- Related: [Innateness, AlphaZero, and Artificial Intelligence (Jan 2018)](https://arxiv.org/abs/1801.05667)
- [Neural Program Synthesis with Priority Queue Training (Abolafia et. al., Jan 2018)](https://arxiv.org/abs/1801.03526)
	- Method: iterative optimisation
		- train RNN on dataset of K best programs from a priority queue of programs generated so far.
		- synthesise new programs and add them to priority queue by sampling from the RNN
	- Results: 'PQT algorithm significantly outperforms (RL and genetic algorithm) baselines on a simple but expressive Turing complete langauge called  BF'
		- Synthesises short, human readable programs (by adding program  length penalty to reward function) 

## Papers to post summaries of (already read)
- [Deep Reinforcement Learning from Human Preferences]

## Posts to read
- [Framework to look at ML research](http://blog.evjang.com/2017/11/exp-train-gen.html?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)

## Papers to maybe-read
- [CycleGAN: A Master of Steganography](https://arxiv.org/abs/1712.02950)
	- CycleGAN can be used to train a generator of adversarial examples
		- it learns to '"hide" information about a soruce image inside the generated image in nearly imperceptible, high-frequency noise'
	- Steganography: 'hiding messages or information within other non-secret text or data.
	- CycleGANS: prev successfully used to learn correspondences between two image distributions
		- train two maps (functions) F:X->Y, G:Y->X in parallel 
			- (image classes X and Y) two functions 
			- goal: satisfy conditions:
				1. Fx ~ p(y) for x~p(x), Gy ~p(x) for y ~ p(y)
				2. GFx = x, FGy = y for all x in X, for all y in Y
		- see intro of this paper or Zhu et al., Mar/Nov 2017](https://arxiv.org/abs/1703.10593)
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation (Karras et. al., Oct 2017)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
	- New training method for GANs that speeds up and stabilises training, which produces images uf 'unprecedented quality' (fake celebrity CelebA images)
- [Applied Machine Learning at Facebook: A Datacenter Infrastructure Perspective (Hazelwood et. al., Dec 2017)](https://research.fb.com/publications/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective/)
	- High performance computing, 
- [Deep Learning Scaling is Predictable, Empirically (Hestness et. al., Dec 2017)](http://research.baidu.com/deep-learning-scaling-predictable-empirically/)
	- How scale relates to performance
	- Results show that 'generalization error—the measure of how well a model can predict new samples—decreases as a power-law of the training data set size'
- [A Flexible Approach to Automated RNN Architecture Generation (Schrimpf et. al., Dec 2017)](https://arxiv.org/abs/1712.07316)
	- Method: 'using a recursive neural network to iteratively predict the performance of new architectures, reducing the need for actual full-blown testing of the models'
	- Result: 
	- Summaries from Import AI #74
- [Recent Advances in Recurrent Neural Networks (Salehinejad et. al., Jan 2018)]
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
](https://arxiv.org/abs/1801.01290)
	- tackles problems with model-free deep RL: (1) high sample complexity, (2) brittle convergence properties -> necessitate meticulous parameter tuning
	- result:
		- state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods
		- stable off-policy method
	- model: combines off-policy updates with a stable stochastic actor-critic formulation
	- Framework aims to succeed at the task while acting as randomly as possible

## Paradigms
- [Deep Neuroevolution (Uber, Dec 2017)](https://eng.uber.com/deep-neuroevolution/)
	- Claim: Neuroevolution, where neural networks are optimized through evolutionary algorithms, is also an effective method to train deep neural networks for reinforcement learning (RL) problems.
	- [[Implementations]](https://github.com/uber-common/deep-neuroevolution)

## Previous papers to print
### General
- [A Deep Hierarchical Approach to Lifelong Learning in Minecraft](https://arxiv.org/abs/1604.07255)
- [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057)
- [An Actor-Critic Algorithm for Sequence Prediction](https://openreview.net/forum?id=SJDaqqveg)
- [End-to-End Online Writer Identification With Recurrent Neural Network](http://ieeexplore.ieee.org/abstract/document/7801018/?reload=true)
- [Gated Orthogonal Recurrent Units: On Learning to Forget](https://arxiv.org/abs/1706.02761)
- [Hierarchical Multiscale Recurrent Neural Networks](https://openreview.net/forum?id=S1di0sfgl)

### Multi-agent RL
- [A Unified Game-Theoretic Approach to
Multiagent Reinforcement Learning](https://arxiv.org/pdf/1711.00832.pdf)

### Capsule Networks
- [Dynamic Routing between Capsules (Capsule networks)](https://arxiv.org/pdf/1710.09829.pdf)
- [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)

## MRes Project
- [FeUdal Networks for Hierarchical Reinforcement Learning (Vezhnevets et. al., 2017)](https://arxiv.org/abs/1703.01161)
- [Reinforcement Learning with Unsupervised Auxiliary Tasks (Jadergberg et. al., 2016)](https://arxiv.org/abs/1611.05397)
- [Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning (Dilokthanakul et. al., Nov 2017)](https://arxiv.org/pdf/1705.06769.pdf)
- [A Deep Hierarchical Approach to Lifelong Learning in Minecraft](https://arxiv.org/abs/1604.07255)
- [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057)
- [RL for complex goals using TF (O'Reilly blog post)](https://www.oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow)

#### Tangentially relevant
- [Hierarchical Multiscale Recurrent Neural Networks](https://openreview.net/forum?id=S1di0sfgl)

#### Friend linked to
- [Learning and Transfer of Modulated Locomotor Controllers (Heess et. al., 2017)](https://arxiv.org/abs/1610.05182)
