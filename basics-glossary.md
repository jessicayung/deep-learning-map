# Basics Glossary


#### Artificial Neural Networks
- Typical Components of a Neural Network
	- Input
	- Score function: maps raw data to class scores (for classification)
		- Layers
		- Conversion to output
	- Loss Function
		- e.g. accuracy?
	- Optimiser (to minimise loss)

#### Convolutional Neural Networks
- 

#### Recurrent Neural Networks

Softmax

#### Backpropagation


- Claim: Many image tasks such as object detection and segmentation can be reduced to image classification

#### Measures of distance

Often used for regularisation.

- L1 distance 
	- d_1(x, y) = \sum_i|x_i - y_i|
- L2 distance 
	- d_2(x, y) = \sqrt{\sum_i (x_i - y_i)^2}


#### Data
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

#### Model performance
- Underfit
- Overfit
- Capacity
- Hyperparameters

#### Things to consider
- Computational cost at train and test

- t-SNE (t-Distributed Stochastic Neighbour Embedding)
	- Dimensionality reduction technique for visualisation of high-dimensional datasets
	- Gives each datapoint a location in a 2D or 3D map.
	- Method:
		- Converts Euclidean distances between datapoints into conditional probabilities that represent similarities
		- TODO: complete
	- [[Website]](http://lvdmaaten.github.io/tsne/) [[Paper]](http://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)