## Neural Networks Components

This is a table that lists components of a neural networks with examples.

<table>
	<th>Category</th><th>Examples</th>
	<tr><td>Data</td><td></td></tr>
	<tr><td>Data Preprocessing</td><td></td></tr>
	<tr><td>Model initialisation</td><td><ul>
		<li>Weight initialisation<ul>
			<li>Glorot</li>
			<li>Gaussian</li>
			<li>Truncated Gaussian</li>
		</ul></li>
		<li>Bias initalisation<ul>
			<li>All zeroes</li>
			<li>Slightly negative</li>
		</ul></li>
	</ul></td></tr>
	<tr><td>Input</td><td></td></tr>
	<tr><td>Input layer</td><td></td></tr>
	<tr><td>Hidden Layers: (Usually linear) op</td><td><ul>
		<li>Linear: WX + b</li>
		<li>Convolutional layer</li>
		<li>RNN layer</li>
	</ul></td></tr>
	<tr><td>Hidden Layers: Activation</td><td><ul>
		<li>ReLU</li>
		<li>Tanh</li>
		<li>Sigmoid</li>
	</ul></td></tr>
	<tr><td>Connections</td><td><ul>
		<li>Skip connections</li>
	</ul></td></tr>
	<tr><td>Regularisation between layers</td><td><ul>
		<li>Dropout</li>
	</ul></td></tr>
	<tr><td>Data preprocessing between layers</td><td><ul>
		<li>Batch normalisation</li>
	</ul></td></tr>
	<tr><td>Output layer: convert hidden layer output to predictions</td><td><ul>
		<li>Linear</li>
		<li>Softmax (classification)</li>
	</ul></td></tr>
	<tr><td>Output</td></tr>
	<tr><td>Loss function: data loss</td><td><ul>
		<li>MSE</li>
	</ul></td></tr>
	<tr><td>Loss function: regularisation loss</td><td><ul>
		<li>L1 norm</li>
		<li>L2 norm</li>
	</ul></td></tr>
	<tr><td>Optimiser</td><td><ul>
		<li>SGD: Stochastic Gradient Descent</li>
		<li>SGD with learning rate tuning:
			<ul>
				<li>Adam</li>
				<li>RMSprop</li>
				<li>Adagrad</li>
			</ul></li>
	</ul></td></tr>
	<tr><td>Parameter update method</td><td><ul>
		<li>For SGD:<ul>
			<li>Vanilla update</li>
			<li>Momentum update</li>
			<ul>
				<li>Nesterov momentum update</li>
			</ul>
		</ul>
		</li>
	</ul></td></tr>
	<tr><td>Gradient calculation</td><td><ul>
		<li>Backpropagation</li>
		<li>Synthetic gradients</li>
	</ul></td></tr>
</table>