# Libraries

- [imgaug (Dec 2017)](https://github.com/aleju/imgaug): Automates process of augmenting an image dataset using transforms
	- [[Code]](https://github.com/aleju/imgaug) [[Docs]](http://imgaug.readthedocs.io/en/latest/)

## Reinforcement Learning
- [Ray RLLib (Dec 2017)](http://ray.readthedocs.io/en/latest/rllib.html)	
	- [[Website]](http://ray.readthedocs.io/en/latest/rllib.html) [[Code]](https://github.com/ray-project/ray/tree/master/python/ray/rllib) [[Paper]](https://arxiv.org/abs/1712.09381)
	- Can implement algorithms by composing and reusing a small number of standard components
	- from UC Berkeley

## Trained Models
- [wav2letter (Jan 2018)](https://github.com/facebookresearch/wav2letter)
	- Facebook's speech recognition system (implemented architecture) with pre-trained models for the Librispeech dataset
		- Applications: transcribing speech
	- [[Code]](https://github.com/facebookresearch/wav2letter) [[Paper: Wav2Letter: an End-to-End ConvNet-based Speech Recognition System (Collobert et. al., Sept 2016)]](https://arxiv.org/abs/1609.03193) [[Paper: Letter-Based Speech Recognition with Gated ConvNets (Liptchinsky et. al., Dec 2017)]](https://arxiv.org/abs/1712.09444)


## Frameworks for distributed AI
- [Ray](http://bair.berkeley.edu/blog/2018/01/09/ray/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)
	- For turning prototype algorithms into high-performance distributed applications
	- Compatible with e.g. TensorFlow, PyTorch, MXNet