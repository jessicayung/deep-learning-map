
# Papers to print

## Next papers to look at
- [Feature Visualisation](https://distill.pub/2017/feature-visualization/)
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

## Papers to post summaries of (already read)
- [Deep Reinforcement Learning from Human Preferences]
- [Gridworlds]

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

#### Tangentially relevant
- [Hierarchical Multiscale Recurrent Neural Networks](https://openreview.net/forum?id=S1di0sfgl)

#### Friend linked to
- [Learning and Transfer of Modulated Locomotor Controllers (Heess et. al., 2017)](https://arxiv.org/abs/1610.05182)