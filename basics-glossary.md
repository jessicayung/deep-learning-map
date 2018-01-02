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
- Gradient check: computing the analytic gradient and comparing it to the numerical gradient to check the correctness of your (analytical) implementation
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


### Convolutional Neural Networks


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
