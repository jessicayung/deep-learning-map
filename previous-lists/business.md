# Business Applications

- [eCommerceGAN : A Generative Adversarial Network for E-commerce (Kumar et. al., Jan 2018)](https://arxiv.org/abs/1801.03244)
	- Generates orders made in e-commerce websites
		- result: high correlation between real customer orders and GAN-generated ones in a 3D t-SNE plot with few outliers
		- implications: [Amazon can simulate 'long tail' of consumer x product combinations -> can optimise supply-chain/just-in-time inventory/marketing campaigns better](https://jack-clark.net/2018/01/15/import-ai-77-amazon-tests-inventory-improvement-by-generating-fake-customers-with-gans-the-imagenet-of-video-arrives-and-robots-get-prettier-with-unity-mujoco-tie-up/)
	- Creates 'dense and low-dimensional' representation of e-commerce orders
	- Trains e-commerce-conditional-GAN (ec^2GAN) to generate plausible orders involving a particular product

- Using VAEs to predict/detect cyberattacks on utilities
	- 