# Basics Glossary


### Artificial Neural Networks
- Typical Components of a Neural Network
	- Input
	- Score function: maps raw data to class scores (for classification)
		- Layers
	- Loss Function (also cost function, objective)
		- e.g. Softmax, SVM
		- to quantify quality of a set of parameters
	- Optimiser (finding parameters to minimise loss)

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
- Scaling input features so values range from [-1,1]

### Measures of distance

Often used for regularisation.

- L1 distance 
	- $d_1(x, y) = \sum_i|x_i - y_i|$
- L2 distance 
	- $d_2(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$

### Loss Functions
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

#### Regularisation
- Loss is usually = Data loss + Regularisation loss
	- Regularisation loss is usually NOT a function of the data, but of the model parameters
- Regularisation: Preference for certain sets of weights over others. 
	- Usually to prevent overfitting or reduce ambiguity when there exist multiple solutions (e.g. when weights $\lambda W$ all yield same output, for positive real $\lambda$).
- L2 norm
	- Discourages large weights
		- 'tends to improve generalization, because it means that no input dimension can have a very large influence on the scores all by itself'
	- Elementwise quadratic penalty over all parameters:
		- $R(W) = \sum_k \sum_l W_{k,l}^2$
			- sums up all squared elements of W
	- Aside: Leads to max margin property in SVMs
- Parameters like scaling of regularisation loss usually determined by cross-validation
- Common to regularise only weights W and not biases b because biases don't control the strength of influence of an input dimensioon, but in practice this often turns out to have a negligible effect.
- If have regularisation loss, cannot achieve zero loss on all examples (assuming examples are distinct), since e.g. for L2 norm, zero loss only possible when W = 0.

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
- In practice, usually use ReLU. Don't use sigmoid.
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
- Maxout


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
