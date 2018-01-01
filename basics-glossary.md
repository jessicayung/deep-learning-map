# Basics Glossary


### Artificial Neural Networks
- Typical Components of a Neural Network
	- Input
	- Score function: maps raw data to class scores (for classification)
		- Layers
		- Conversion to output
	- Loss Function (also cost function, objective)
		- e.g. accuracy?
	- Optimiser (to minimise loss)

### Convolutional Neural Networks


### Recurrent Neural Networks


### Backpropagation


### Measures of distance

Often used for regularisation.

- L1 distance 
	- $d_1(x, y) = \sum_i|x_i - y_i|$
- L2 distance 
	- $d_2(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$


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

### Loss Functions
- Multiclass Support Vector Machine (SVM) loss
	- Wants correct class to have score higher than incorrect classes by at least some fixed margin $\Delta$.
	- $L_i = \sum_{j\ne y_i}\max(0,s_j-(s_{y_i}-\Delta))$ 
		- where $s_j$ is the score for the jth class,
		- i is the index for examples
	- Aside: can be formulated in other ways (OVA, AVA)
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
