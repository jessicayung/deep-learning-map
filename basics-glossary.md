# Basics Glossary

Most of these definitions are paraphrased from [CS231n](http://cs231n.github.io/).

### Artificial Neural Networks
- 'Perform sequences of linear mappings with interwoven non-linearites'
- Typical Components of a Neural Network
	- Input
	- Score function: maps raw data to class scores (for classification)
		- Layers
		- Output layer
			- Usually don't have an activation fn
	- Loss Function (also cost function, objective)
		- e.g. Softmax, SVM
		- to quantify quality of a set of parameters
	- Optimiser (finding parameters to minimise loss)
- Size: usually measured by number of parameters
	- Small networks not preferred: they are harder to train with local methods such as gradient descent.
		- Many minima of loss functions are bad and easy to converge to -> variance in final loss
	- Large networks: beware of overfitting. 
		- Tackle using regularisation.
- Representational power: NNs with at least one hidden layer are universal approximators
	- but this has little to do with their widespread use
- Number of layers
	- (Fully Connected only) In practice, going beyond 3 layers rarely helps much more.
	- CNNs: depth has been found to be extremely important (10+ learnable layers).

### Data
- Training set
	- Data used to directly train your model
- Validation set
	- Data used to tune hyperparameters
		- Usually choose hyperparameters with lowest validation error
	- Like a fake test set
- Test set
	- Data used to measure generalisation of your model: used only once!
- Generalisation
- K-fold Cross-validation
	- Iterate over k different validation sets and average performance across them
	- Especially useful if training set is small (e.g. < 1000 exampless).
	- BUT computationally expensive
	- Usually K = 3, 5 or 10

### Model performance
- Underfit
- Overfit
- Capacity
- Hyperparameters

### Things to consider
- Computational cost at train and test

### Data preprocessing
- Centering data by subtracting mean from each feature
	- `X-= np.mean(X, axis=0)`
	- For images, can subtract a single value from all pixels for convenience: `X -= np.mean(X)`.
- Normalisation
	- Method (2 ways)
		- 1: Divide each zero-centered feature by its standard devation:
			- `X /= np.std(X, axis=0)`
		- 2: Scaling input features so values range from [-1,1]
	- 'Only if different features have different scales but should be of approximately equal importance to the learning algorithm' -> hm
- PCA (principle component analysis)
	- Dimensionality reduction: keep K dimensions of data that contain the most variance.
	- Method:
		1. Center data at zero (as above).
		2. Compute covariance matrix
		3. Compute SVD factorisation of data covariance matrix: eigenvectors U, singular values S, V.
		4. Project original (zero-centered) data into the eigenbasis
		5. Use only the top K eigenvectors (reduce to K dims).
- Whitening: 
	- Used after PCA. Normalises the scale of the data.
	- Method: Take the data in the eigenbasis and divide each dimension by the egienvalue to normalise the scale
	- Cons: Can exaggerate noise (higher frequencies) since it stretches all dims to be of equal size in the input
		- Can mitigate with stronger smoothing
- Note that statistics such as the mean should only be computed from training data.
- PCA and whitening rarely used with CNNs.

### Weight initialisation
- Initialise to small random numbers
	- Not too small for deep nets: Smaller weights mean smaller gradients which could reduce the 'gradient signal' flowing backward through a network and so be a problem for deep networks (because there are so many layers it needs to propagate through)
- Normalise variance per neuron by 1/sqrt(num_input)
	- Alts:
		- For NNs with ReLU neurons: `sqrt(2.0/n)` [(He et. al.)](https://arxiv-web3.library.cornell.edu/abs/1502.01852)
			- **current recommendation**
		- `sqrt(2.0/(n_in + n_out))` [(Glorot et. al.)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
	- Variance of dist of outputs from a randomly initialised neuron grows with number of inputs. 
	- 1/sqrt(num_inputs) is the recommended heuristic scaling factor 
	- So all neurons have approx same output dist, empirically improves rate of convergence
	- Fan-in: number of inputs
- Alt: sparse initialisation
	- Set all weight matrices to zero (but say connections sampled from small gaussian? TODO: follow up), but 
	- Break symmetry by randomly connecting each neuron to a fixed number of neurons below it, e.g. 10 neurons.
- Bias initialisation
	- Commonly initialise to zero
	- (Unclear if provides consistent improvement) ReLU non-linearities: can init to small constant like 0.01 to ensure all ReLU units fire in the begining and thus obtain and propagate some gradient.
- Mistakes
	- All-zero initialisation (all weights the same): no asymmetry between neurons, all compute same gradients and undergo same parameter updates. 
- Batch normalisation 
	- Make nets more robust to bad initialisation.
	- Idea: Explicitly forces activations throughout net to take on unit gaussian dist at start of training.
		- Possible because normalisation is a simpe differentiable operation.
		- Like doing preprocessing at each layer of the net.
	- Method:
		- Insert BatchNorm laye immediately after fully connected layers.
	- [Paper by Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167)

### Measures of distance

Often used for regularisation.

- L1 distance 
	- $d_1(x, y) = \sum_i|x_i - y_i|$
- L2 distance 
	- $d_2(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$

### Loss Functions (Data loss)
- Loss is usually = Data loss + Regularisation loss
- Loss usually mean loss per example over training data examples
- Multiclass Support Vector Machine (SVM) loss
	- Wants correct class to have score higher than incorrect classes by at least some fixed margin $\Delta$.
	- $L_i = \sum_{j\ne y_i}\max(0,s_j-(s_{y_i}-\Delta))$ 
		- where $s_j$ is the score for the jth class,
		- i is the index for examples
	- Aside: can be formulated in other ways (OVA, AVA)
	- Convex function, non-differentiable (due to kinks) but [subgradient](https://en.wikipedia.org/wiki/Subderivative) still exists and is commonly used instead.
		- Subgradient: set of slopes of lines drawn through a point f(x0) which is everywhere either touching or below the graph of f.
- Cross-entropy loss
	- Minimises cross-entropy between estimated class probabilities and the distribution where all p(x) mass is on the correct class
	- $L_i = -\log(\frac{e^{f_{y_i}}{\sum_j e^{f_j}})$
		- equivalently: $L_i = -f_{y_i} + \log\sum_j e^{f_j}$
		 - where $f_j$ = jth element of vector of class scores $f$.
	- Problems:
		- If number of classes is very large (e.g. ImageNet 22k classes), may be helpful to use hierarchical softmax
			- decomposes labels into tree (each label is a path), softmax classifier trained at each node of tree.
			    - Structure of tree strongly affects performance as is usually problem-dependent
	- More Theory
		- Equivalent to minimising Kullback-Leibler divergence $D_{KL}$ between the two distributions. 
		 	- since $H(p,q) = H(p) + D_{KL}(p||q)$, and $H(p) = 0$.
		- Information Theory Cross Entropy
		 	- $H(p,q)=-\sum_xp(x)\log q(x)$
		 		- where p is a 'true' distribution and q is an estimated distribution
	- Probabilistic Interpretation
		- Interpret $P(y_i|x_i;W) = \frac{e^{f_{y_i}}{\sum_j e^{f_j}}$ as the 
			- Normalised probability assigned to the correct label $y_i$ given the image $x_i$ and parameterised by $W$.
			- since softmax classifier interprets $f$ scores as unnormalised log probabilities.
		- i.e. minimising negative log likelihood of the correct class, i.e. performing Maximum Likelihood Estimation (MLE).
		- Can thus also interpret R(W) as 'coming from a Gaussian prior over the weight matrix W, where we are performing Maximum a posteriori (MAP) estimation'
		- then the cross-entropy loss $L_i = -\log P(y_i|x_i;W)$.
- Softmax function
 	- $f_j(z) = \frac{e^{z_j}{\sum_k e^{z_k}}$
 	- 'Takes vector of arbitrary real-valued scores in z and squashes it to a vector of values between zero and one that sum to one.'
 	- Problems: Dividing large numbers (exponentials may be large) may be numerically unstable, so multiply top and bottom by constant C, where e.g. $\log C = - \max_j f_j$
 		- i.e. shift values inside vector $f$ so highest value is zero.
- Softmax classifier notes
	- Outputs unnormalised log probabilities whose peakiness depends on regularisation strength.
		- Higher regularisation strength -> less peaky
- L2 vs Softmax
	- Softmax more stable, L2 harder to optimise
- Structured loss
	- Case where labels can be arbitrary structures such as graphs, trees or other complex objects. Space of structured assumed to be large and not easily enumerable.
	- Idea: Deand margin between correct structure y_i and highest-scoring incorrec structure.
	- Usually devise special solvers (as opposed to gradient descent) that exploit simplifying assumptions of the structure space.
- For regression
	- L2 norm squared (of the difference between the prediction quantity and true answer)
		- i.e. $L_i = ||f-y_i||^2_2$
		<!-- TODO: expand this -->
	- L1 norm of the difference between the prediction quantity and true answer
		- $L_i = ||f-y_i||_1 = \sum_j|f_j-(y_i)_j|$
	    - L2 norm squared because gradient becomes 
	    simpler_

### Regularisation loss
- Loss is usually = Data loss + Regularisation loss
    - Regularisation loss is usually NOT a function of the data, but of the model parameters
- Regularisation: Preference for certain sets of weights over others. 
	- Usually to prevent overfitting or reduce ambiguity when there exist multiple solutions (e.g. when weights $\lambda W$ all yield same output, for positive real $\lambda$).
- L2 norm
	- Elementwise quadratic penalty over all parameters:
		- $R(W) = \sum_k \sum_l W_{k,l}^2$
			- sums up all squared elements of W
		- often $\frac{1}{2}\lambda w^2$, 0.5 because then gradient wrt w is $\lambda w$.
	- Discourages large weights
		- 'tends to improve generalization, because it means that no input dimension can have a very large influence on the scores all by itself'
		- encourages more diffuse (vs peaky) vectors
	- During gradient descent param update, each weight is decayed linearly towards zero. `W+= 
	-lambda * W`.
	- Aside: Leads to max margin property in SVMs
- L1 regularisation
	- $R(W) = \lambda|w|$
	- Makes weight vectors sparse during optimisation
		- Neurons end up using only a sparse subset of most important inputs and become nearly invariant to 'noisy' inputs.
- ElasticNet regularisation
	- Combine L1 with L2 regularisation: $R(W) = \lambda_1|w| + \lambda_2w^2$
- Notes (L1, L2):
	- Claim: L2 expected to give superior performance over L1 if we're not concerend with explicit feature selection (CS231n)
	- Common to regularise only weights W and not biases b because biases don't control the strength of influence of an input dimensioon, but in practice this often turns out to have a negligible effect.
	- If have regularisation loss, cannot achieve zero loss on all examples (assuming examples are distinct), since e.g. for L2 norm, zero loss only possible when W = 0.
- Max norm constraints: 
	- Absolute upper bound on magnitude of weight vector for each neuron
	- Method: clamp weights after parameter updates
		- called projected gradient descent? TODO: check.
	- Pros: Network cannot 'explode' even when learning rate is set too high because updates always bounded.
- Dropout
	- Only keep a neuron active with probability p (hyperparameter), set it to zero otherwise
	- [Paper by Srivastava et. al., 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
	- NOTE for raw implementations: usually scale by 1/p AND drop during train time to keep the expected output of each neuron the same during test and train
- DropConnect (not common)
	- Random set of weights set to zero during forward pass
- Regularising bias (not common)
	- But rarely leads to significantly worse performance
- Parameters like scaling of regularisation loss usually determined by cross-validation
	
#### Helper functions in loss functions
- Hinge Loss
	- $f(x) = max(0, x)$
	- Squared Hinge Loss: $max(0, -)^2$
		- Penalises violated margins more strongly. Less common.

### Optimisation
- Key hyperparameter: learning rate (step size of gradient descent)
- Centered difference formula
	- $[f(x+h)-f(x-h)]/2h$ 
	- in practice often better for computing numeric gradient than the typical gradient formula ($[f(x+h)-f(x)]/h$).
	- Twice as expensive but more precise (Error terms on order of $O(h^2))$ (second order approximation) vs typical formula with error on order of $O(h)$ (first order approximation).
- Gradient check: computing the analytic gradient and comparing it to the numerical gradient to check the correctness of your (analytical) implementation
	- Consider the relative error $\frac{|f'_a - f'_n|}{\max(|f'_a|, |f'_n|)}$
		- AND explicitly keep track of the case where both are zero and pass the gradient check in that edge case.
		- Empirical figures for relative error (CS231n):
			- > 1e-2: gradient probably wrong
			- between (1e-2, 1e-4): uncomfortable
			- under 1e-4: usually okay for objectives with kinks, but too high otherwise
			- under 1e-7: good
		- Will get higher relative errors with deeper networks
	- Use double precision
	- More at [CS231n](http://cs231n.github.io/neural-networks-3/)
- Gradient descent:
	- repeatedly evaluating the gradient and then performing a parameter update (`weights += step_size *  weights_grad`)
- Mini-batch gradient descent:
	- Parameter update after computing gradient over a batch (subset) of the training data 
	- Works because examples in the training data are correlated
- Stochastic gradient descent
	- Mini-batch size = 1. 
	- In practice people often call mini-batch gradient descent SGD.
<!-- TODO: implement MGD etc -->


### Backpropagation
- TODO: add implementation examples and links to those examples
- Staged computation: breaking up functions into modules for which you can easily derive local gradients, and then chaining them with the chain rule.
	- Group together expressions as a gate (e.g. sigmoid) for convenience 
- Practical notes
    - Useful to cache forward pass variables
- Multiply gate:
	- (two arguments) Local gradients are input values switched then multiplied by output gradient
	- So scale of gradients is directly proportional to scale of inputs.
	- So preprocessing matters: if input too large, need to lower learning rate to compensate.

### Parameter updates
- Using gradients computing via propagation to perform parameter updates
- SGD
	- Vanilla update
		- Change parameters along negative gradient direction (to minimise loss)
		- `x+= - learning_rate * dx`
	- Momentum update
		- ```
		v = mu * v - learning_rate * dx # integrate velocity
		x += v # integrate position
		```
			- 'Momentum' parameter mu is more like the coefficient of friction.
				- Damps velocity and reduces kinetic energy of system, else particle would never come to a stop at the bottom of a hill.
				- Usually set to [0.5, 0.9, 0.95, 0.99] when cross-validated.
			- Optimisation can sometimes benefit a little from momentum schedules, where momentum is increased in later stages of learning.
				- E.g. init 0.5, anneal to 0.99 over multiple epochs.
		- 'Parameter vector will build up velocity in any direction than has consistent gradient' <!-- TODO: ? -->
 		- Physics perspective:
			- Loss as height of hilly terrain, proportional to potential energy U = mgh
			- Initialising parameters with random numbers seen as equivalent to setting a particle with zero initial velocity at some location
			- Optimisation process equivalent to process of simulating parameter vector (particle) as rolling on the landscape
			- Force on particle $F = -\grad U$, so force felt by particle is negative gradient of loss.
			- F = ma, so negative gradient is proportional to acceleration of particle.
		- VS SGD: Gradient only directly influences the velocity, which in turn has an effect on the position. VS SGD gradient directly integrates the position.
	- Nesterov momentum update
		- ```
		v_prev = v
		v = mu + v - learning_rate * dx # velocity update same
		x += -mu * v_prev + (1+mu) * v # position update changes form
		```
		- Idea: Compute gradient at lookahead position `x + mu * v` instead of old position `x`, since momentum alone pushes particle to lookahead position.
		- Stronger theoretical convergence guarantees for convex functions
		- In practice consistently works slightly better than standard momentum
- Second order methods (not common in practice)
	- $x \leftarrow x - [Hf(x)]^{-1}\grad f(x)$.
		- $Hf(x)$ being the Hessian matrix, a square matrix of second-order partial derivatives of the function (describes local curvature of loss fn)
		- No learning rate hyperparameters
		- Impractical for most DL applications because computing Hessian is costly in space and time
		    - So people have developed quasi-Newnton methods that approximate the inverted Hessian, e.g. L-BFGS.
	    - But naive L-BFGS must be computed over entire training set. Getting L-BFGS to work on mini-batches is tricky.

### Annealing learning rate
- Learning rate decay
	- In practice find step decay slightly preferable because hyperparameters are more interpretable.
	- Better to err on the side of slower decay and train for longer if you can afford the computational budget
	- Step decay: reduce lr by some factor (e.g. 0.5) every k epochs
		- Heuristic: reduce lr by a constant whenever the validation error stops improving
	- Exponential decay `lr = init_lr*exp(-k*t)`
		- init_lr, k: hyperparameters
		- t: iteration number (or epoch)
	- 1/t decay `lr = init_lr/(1+k*t)`
		- init_lr, k: hyperparameters
		- t: iteration number

#### Methods to tune learning rate
- Adagrad
	- ```
	# Assume the gradient dx and parameter vector x
	cache += dx**2
	x += - learning_rate * dx / (np.sqrt(cache) + eps)
	``` <!-- TODO: what is cache's init value? -->
		- cache: used to normalise parameter update step element-wise. Weights with higher gradients have effective lr reduced and vice versa
		- eps: smoothing term (1e-8 to 1e-4) that avoids division by zero
	- Cons: monotonic learning rate usually proves too aggressive and stops learning too early
	- By Duchi et. al.
- RMSprop
	- ``` cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps) # same as Adagrad
	```
		- Adjusted Adagrad (to reduce aggressive, monotonically decreasing learning rate -> has equalizing effect but upds do not get monotonically smaller) 
			- Uses moving average of squared gradients instead of squared gradients.
		- `decay_rate`: hyperparameter, typical values [0.9, 0.99, 0.999]
	- Very effective
- Adam
	- ```
	# Simplified version
	m = beta1*m + (1-beta1)*dx
	v = beta2*v + (1-beta2)*(dx**2)
	x += - learning_rate * m / (np.sqrt(v) + eps)
	```
		- Smooth version of gradient (m) used rather than raw, possibly noisy gradient vector `dx`.
		- Full version includes bias correction:
			- Corrects for fact that in the first few timesteps m, v both initialised and therefore biased at zero)
			- ```
			# t is your iteration counter going from 1 to infinity
			m = beta1*m + (1-beta1)*dx
			mt = m / (1-beta1**t)
			v = beta2*v + (1-beta2)*(dx**2)
			vt = v / (1-beta2**t)
			x += - learning_rate * mt / (np.sqrt(vt) + eps)
```
	- Like RMSprop with momentum
	- Currently recommended as default algorithm to use

### Activation functions
- In practice, usually use ReLU. Be careful with learning rates, possibly monitor fraction of 'dead' units in a network. Don't use sigmoid.
- ReLU $f(x)=\max(0,x)$
	- Pros:
		- Greatly accelerates the converges of SGD compared to the sigmoid/tanh functions, possibly due to linear, non-saturating form (Krizhevsky et. al.).
		- Cheaper computationally than tanh, sigmoid
	- Cons:
		- ReLU units can irreversibly 'die' during training: a large gradient flowing through a ReLU neuron could cause weights to update such that the neuron will never activate on any datapoint again.
			- so gradient through that unit will always be zero.
			- Tackle by e.g. **decreasing learning rate**.
- Leaky ReLU: $f(x) = 1(x<0)(\alpha x) + 1(\geq 0)(x)$, where $\alpha$ is a small constant.
	- Attempt to fix 'dying ReLU' problem. When x < 0, function has small negative slope instead of being zero.
	<!-- TODO: verify through differentiating -->
	- Consistency unclear
- Sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$
	- $\frac{d\sigma(x)}{dx} = (1-\sigma(x))\sigma(x)$
	- Rarely used now because
		- Sigmoids saturate and kill gradients: when neuron's activation saturates at 0 or 1, the gradient at these regions is almost zero
			- So almost no signal will flow through the neuron to its weights and thus to its data
			- Also need to be careful that initial weights don't saturate at 0 or 1 
		- Sigmoid outputs are not zero-centered (less serious)
			- -> neurons in later layers would be receiving data that is not zero-centered
			- If data coming into neurons is always positive, gradients on weights during backprop will either always be positive or always be negative.
			- May introduce zigzagging dynamics in gradient updates
			- BUT once gradients summed across a batch, 'the final update for the weights can have variable signs, somewhat mitigating this issue' <!-- TODO: =? Also add code examples for these two points -->
- Tanh
	- Squashes real number to range [-1,1]
	- Activations saturate, but output is zero-centered. 
		- So tanh always preferred to sigmoid.
		- Tanh is scaled sigmoid neuron: $\tanh(x) = 2\sigma(2x) - 1$
- Maxout $\max(w_1^Tx+b_1, w_2^Tx+b_2)$
	- Generalisation of ReLU and Leaky ReLU
	- Pros & Cons
		- Has benefits of ReLU (linear, no saturation) without drawbacks (dying ReLU).
		- BUT doubles number of parameters for each neuron, leading to a high total number of paramaters
	- Introduced by Goodfellow et. al.

### Before learning (sanity checks)
- Check you're getting the loss you expect when initialising with small parameters
	- Check data loss alone
		- e.g. CIFAR-10 with Softmax classifier expect initial loss to be -ln(0.1) = 2.302
		- e.g. Weston Watkins SVM expect all desired margins to be violated, so expected loss = 9
	- Then increasing regularisation strength should increase loss
- Overfit a tiny subset of data
	- e.g. 20 examples, make sure you can achieve zero cost
	- Set regularisation = 0 for this.

### During learning: things to monitor
- Loss against epochs:
	- if it goes flat (or increases quickly), learning rate likely too high
	- if linearly decreasing, learning rate likely too low
	- Variation between epochs increases as batch size decreases
- Training and validation accuracy
- Ratio of magnitudes of updates:weights (magnitudes)
	- update_scale/weights_scale should be around 1e-3 (heuristic)
		- magnitude = norm of vector
		- If lower, learning rate might be too low and vice versa 
	- alt: keep track of norm of gradients and gradient updates, usually correlated and give approximately the same results
- Activation / Gradient distributions per layer
	- Diagnosing incorrect initialisation
		- Can identify all neurons outputting zero or all neurons being completely saturated at either -1 or 1.
	- Method: Plot activation / gradient histograms: should not see strange distributions
- (For images) First-layer visualisations

### Key hyperparameters to tune
- Initial learning rate
- Learning rate decay schedule (e.g. decay constant)
- Regularisation strength (L2 penalty, dropout strength)

### Tips for hyperparameter optimisation
- One validation fold (of respectable size) vs CV simplifies the code base
- Search for hyperparameters on log scale. E.g. `learning rate = 10 ** uniform(-6,1)`. Similar for regularisation strength, since these have multiplicative effects on the training dynamics.
	- vs dropout usually searched in original scale
- Prefer random search to grid search ([Bergstra and Bengio](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf))
	- Often the case that some hyperparameters matter much more than others
- Check best values are not on order (else more optimal settings may be outside your interval)
- First search coarse search (and with fewer epochs, e.g. 1 epoch), then move to finer search
	- Some hyperparameter settings can lead the model to not learn at all, or immediately explode with infinite cost
- Bayesian Hyperparameter Optimisation: 
	- Algorithms to more efficiently navigate the hyperparameter space via exploration-exploitation tradeoff
	- e.g. Spearmint, SMAC, Hyperopt. 
	- In practice (for CNNs) it's hard to beat random search in carefully-chosen intervals

### Ensembles
- In practice training independent models and averaging predictions at test time is a reliable approach to improve performance of NNs
- Approaches to forming an ensemble
	- Same model, different initialisations (with hyperparameters determined by CV)
	- Top few hyperparameter configurations
	- Different checkpoints of a single model
	- Running (exponentially decaying) average of weights used during training
		- Intuition: Objective is bowl-shaped, network is jumping around mode, so average has higher chance of being somewhere near the mode.

### Convolutional Neural Networks
- Properties
	- Constrain architecture assuming input consists of images (or similarly structured data)
		- inputs and outputs are 3D volumes (width, height, depth of image)
        - Motivation: regular neural nets don't scale well to full images
    - Shared weights: reduce number of parameters
- Convolutional Layer
	- Filters: Each filter is smaller than image along width, height dimensions, depth same as image.
	- Operation: Convolve (// slide) filter across width and height of input and compute dot products between entries of filter and input at every position
	- Output: Produce 2D activation map for each depth layer -> stack them to produce output
	- Connectivity
		- Locally (vs fully) connected: connect each neuron to only a local region of the input volume. 
			- (i.e. Sparse connectivity: only connect each neuron to a subset of neurons in the previous layer)
		- Parameter: receptive field of neuron (filter size/dimensions)
	- Spatial arrangement
		- Depth of output volume (Number of filters)
			- Each filter is meant to learn to look for something different in the input, e.g. presence of various oriented edges or blobs of colour.
			- Depth column / fibre: Set of neurons all looking at the same region of the input
		- Stride: Number of pixels we move at a time when we slide the filter around the image
			- Usually 1 or 2
			- Larger stride produces smaller output spatially (width, height)
		- Padding
			- Zero padding: pad input volume with zeros around the border.
				- To control size of output volume, usually to keep it the same dim as the input volume
				- Hyperparameter: size of zero padding
		- Output spatial size (along each of width, height): (W-F+2P)/S + 1, where
			- W = input vol size
			- F = filter size
			- S = stride
				- If (W-F+2P) is not divisible by S -> depends how yo uhandle it, e.g. can floor (crop image) or throw error
			- P = padding (on one side)
		- Weight sharing
			- Weights for filter are the same within one neuron.
				- so forward pass can be computed as a convolution of neuron's weights with input volume
			- Assume that 'if one feature is useful to compute at some spatial position (x1,y1), then it should also be useful to compute at a different position (x2,y2).'
				- Locally-connected layer: relax parameter sharing scheme since it may not make sense if e.g. input images to ConvNet have specific centered structure
		- im2col implementation (matrix multiplication)
			- Method:
				- Matrix `X_col`: convert each filter-sized region into a column vector.
				- Matrix `W_row`: convert each filter (depth slice) into a row vector
				- Multiply `X_col` with `W_row` (e.g. `np.dot(X_col, W_row)`) and reshape it to proper output dimensions
			- Cons: Memory-intensive (some values replicated in `X_col`)
			- Pros: There are very efficient implementations of matrix multiplication we can use (e.g. BLAS API)
		- Backpropagation: Also a convolution. TODO: derive
		- 1x1 convolution
			- Makes sense because we operate over 3D and filters always extend through full depth of input volume (TODO: link and elaborate)
		- Dilated convolutions
			- Spaces in between each cell in the filter
			- Allows you to merge spatial information across inputs more aggressively since receptive field grows much quicker
- Pooling Layer
	- Resizes (downsamples) each depth slice of input using MAX operation (Max-pooling)
		- Other less common pooling operations: Average pooling or L2-norm pooling
	- Motivation: 'To reduce amount of parameters and computation in the network, and hence to also control overfitting'
	- Method
		- Hyperparameters: 
			- spatial extent F (filter width, height)
			- stride S
		- Output of size W2 = (W1 - F)/S + 1 (same for height)
			- where W1 is input width, W2 is output width
		- Common parameterisation: 2x2 filters, stride of 2 (discards 75% of activations)
	- Introduces zero parameters
	- Backpropagation: 
	- Looking ahead:
		- [Discarding pooling (2015)](https://arxiv.org/abs/1412.6806) 
		- Discarding pooling layers has been found to be important in training good generative models like variational autoencoders and genarative adversarial networks.
- Normalisation Layer
	- Fallen out of favour because in practice their contribution has been shown to be minimal
- Converting Fully connected layer to Conv layer
	- TODO: add, see [cs231n](http://cs231n.github.io/convolutional-networks/)
- Common ConvNet architectures
	- INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
		- where usually N <= 3, K < 3. N, M, K all non-negative.
		- Two CONV layers before every POOL layer 'generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the deestructive pooling operation'
- Heuristics
	- Prefer a stack of small filter CONVs to one large receptive field CONV layer (that have identical spatial extent).
		- Stack of small filters contain non-linearities that make features more expressive
		- Uses fewer parameters
		- BUT might need more memory to hold intermediate CONV layer results for backprop
	- CNN Layer Sizing Patterns
		- Input layer: multiples of powers of 2 (e.g. 32 (CIFAR-10), 64, 96 (STL-10), 224 (ImageNet), 384, 512)
		- Conv layers: 
			- small filters (3x3 to 5x5)
			- stride = 1
				- smaller strides work better in practice and allow us to leave all down-sampling to POOL layers
				- may have to compromise (increase stride) due to memory constraints, usually compromise only input layer.
			- pad with zeros such that spatial dimensions of input remains the same
				- padding also improves performance, else info at borders 'washed away' too quickly.
		- Pool layers
			- Most common: max-pooling with 2x2 receptive fields, stride=2 (discards 75% of activations)
			- rare to see F > 3 because then pooling 'is then too lossy and aggressive and usually leads to worse performance'.
- Looking ahead
	- Paradigm of linear list of layers has recently been challenged in Google's Inception architectures and Residual Networks from Microsoft Research Asia
	- Hinton's Capsule Networks
- In practice
	- Use whatever works best on ImageNet. Download a pretrained model and finetune it on your data.
- Famous ConvNet architectures
	- LeNet
	- AlexNet (ImageNet 2012 winner)
	- ZF Net (Zeiler and Fergus, ILSVRC 2013 winner)
	- GoogLeNet (Szegedy et. al., ILSVRC 2014 winner)
		- Developed 'Inception Module' that dramatically reduces parameters in the network (4M from AlexNet's 60M)
		- Uses average pooling instead of fully connected layers at the top of the ConvNet
			- Eliminates a large number of parameters that don't seem to matter much
		- Followed up by Inception-v4 etc.
	- VGGNet (Simonyan and Zisserman, ILSVRC 2014 runner-up)
		- Showed that the depth of the network is a critical component for good performance.
			- 16 layers
		- Homogenous architecture that only performs 3x3 convs and 2x2 pooling from beginning to end.
		- Cons: More expensive to evaluate, uses more memory and parameters (140M params, mostly in first FC layer)
			- since found that first FC layers can be removed with no performance downgrade, greatly reducing number of parameters (first FC layer has 100M parameters)
	- ResNet (Residual Network, Kaiming He et. al., winner of ILSVRC 2015)
		- Skip connections
		- Heavy use of batch normalisation
		- No fully connected laters at the end
		- State-of-the-art as of May 10, 2016
- Computational Considerations
	- Memory bottleneck
		- Many modern GPUs have limits of 3/4/6 GB memory
		- Memory sources:
			- Activations
				- Raw activations (forward pass values) and gradients at each layer of a convnet. Kept because they are needed for backpropgation
					- (TODO: could in principle reduce this...? see [cs231n page](http://cs231n.github.io/convolutional-networks/))
			- Parameters
				- Network parameters, their gradients during backpropagation, commonly also a step cache if optimisation is using momentum, Adagrad or RMSprop.
				- Estimate: num params x 3
			- Misc. memory
				- e.g. image data batches and possibly their augmenetd verisons
		- Estimating memory: number of values x 4 (bypes) / 1024**3 (GB).
		- Decrease batch size to make things fit (since most memory usually consumed by activations)

### Recurrent Neural Networks


### Other techniques
- Softmax
	- Binary logistic regression generalised to multiple classes
	- Outputs normalised class probabilities
	- TODO: where should I put this? Loss? 'final-step classifier'?
- t-SNE (t-Distributed Stochastic Neighbour Embedding)
	- Dimensionality reduction technique for visualisation of high-dimensional datasets
	- Gives each datapoint a location in a 2D or 3D map.
	- Method:
		- Converts Euclidean distances between datapoints into conditional probabilities that represent similarities
		- TODO: complete
	- [[Website]](http://lvdmaaten.github.io/tsne/) [[Paper]](http://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)

### Other CS231n notes (Temporary)
- Claim: Many image tasks such as object detection and segmentation can be reduced to image classification
- Linear Classifiers
	- Linear classifier Wx+b is effectively running K (number of classes) classifiers at the same time.
	- Interpreting linear classifiers as template matching: each row of W corresponds to a template/prototype for one of the classes. Score of classes obtained by comparing each template with image using inner product (dot product)
		- Distance: Negative inner product
- Bias trick (to represent W, b as one)
	- Extend input vector x_i by one additional dimension than holds constant 1 (default bias dimension)
	- W, b merged into new W.
- SVM vs Softmax classifiers
	- SVM interprets scores as class scores, loss function encourages correct class to have a score higher by a margin than other class scores
	- Softmax interprets scores as unnormalised log probabilities for each class, encourages log probability of correct class to be high.
	- SVM and Softmax losses are thus not comparable
	- SVM more 'local': happy if score for correct class is sufficiently higher than other classes, vs softmax never fully satisfied

### Other terms
- Subgradient

### References
- [CS231n](http://cs231n.github.io/)

### Other resources
- [Machine Learning for Humans: plain-English explanations of ML](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12)
	- Caveat: I haven't read this yet
- [Jeff Dean's Lecture for YC AI](https://blog.ycombinator.com/jeff-deans-lecture-for-yc-ai/)
- [Heroes of Deep Learning interview series by Andrew Ng](https://www.youtube.com/watch?v=-eyhCTvrEtE&list=PLfsVAYSMwsksjfpy8P2t_I52mugGeA5gR)