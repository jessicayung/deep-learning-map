## Neural Networks Components

This is a table that lists components of a neural networks with examples.

<table>
	<th><td>Category</td><td>Examples</td></th>
	<tr><td>Data</td><td></td></tr>
	<tr><td>Data Preprocessing</td><td></td></tr>
	<tr><td>Model initialisation</td><td><ol>
		<li>Weight initialisation<ol>
			<li>Glorot</li>
			<li>Gaussian</li>
			<li>Truncated Gaussian</li>
		</ol></li>
		<li>Bias initalisation<ol>
			<li>All zeroes</li>
			<li>Slightly negative</li>
		</ol></li>
	</ol></td></tr>
	<tr><td>Input</td><td></td></tr>
	<tr><td>Input layer</td><td></td></tr>
	<tr><td>Hidden Layers: (Usually linear) op</td><td><ul>
		<li>Linear: WX + b</li>
		<li>Convolutional layer</li>
		<li>RNN layer</li>
	</ul></td></tr>
	<tr><td>Hidden Layers: Activation</td><td><ol>
		<li>ReLU</li>
		<li>Tanh</li>
		<li>Sigmoid</li>
	</ol></td></tr>
	<tr><td>Connections</td><td><ol>
		<li>Skip connections</li>
	</ol></td></tr>
	<tr><td>Regularisation between layers</td><td><ol>
		<li>Dropout</li>
	</ol></td></tr>
	<tr><td>Data preprocessing between layers</td><td><ol>
		<li>Batch normalisation</li>
	</ol></td></tr>
	<tr><td>Output layer: convert hidden layer output to predictions</td><td><ol>
		<li>Linear</li>
		<li>Softmax (classification)</li>
	</ol></td></tr>
	<tr><td>Output</td></tr>
	<tr><td>Loss function: data loss</td><td><ol>
		<li>MSE</li>
	</ol></td></tr>
	<tr><td>Loss function: regularisation loss</td><td><ol>
		<li>L1 norm</li>
		<li>L2 norm</li>
	</ol></td></tr>
	<tr><td>Optimiser</td><td><ol>
		<li>SGD: Stochastic Gradient Descent</li>
		<li>SGD with learning rate tuning:
			<ol>
				<li>Adam</li>
				<li>RMSprop</li>
				<li>Adagrad</li>
			</ol></li>
	</ol></td></tr>
	<tr><td>Parameter update method</td><td><ol>
		<li>For SGD:<ol>
			<li>Vanilla update</li>
			<li>Momentum update</li>
			<ol>
				<li>Nesterov momentum update</li>
			</ol>
		</ol>
		</li>
	</ol></td></tr>
	<tr><td>Gradient calculation</td><td><ol>
		<li>Backpropagation</li>
		<li>Synthetic gradients</li>
	</ol></td></tr>
</table>