# Datasets

Note: This page is currently more a list than an indicator of which are the 'main datasets' in research. (13 Jun 2018: working on adding main datasets and converting format to tables)

## Speech
- [Common Voice (Mozilla)](https://voice.mozilla.org/data)
	- 400 hours of speech from 400k recordings from 20k people.

## Robotics
- [RoboCupSimData](https://bitbucket.org/oliverobst/robocupsimdata/)
	- Over 180 hours of robot soccer gameplay.
	- [[Paper]](https://arxiv.org/abs/1711.01703) [[Data]](https://bitbucket.org/oliverobst/robocupsimdata/)

## Self-Driving Cars
<table>
	<tr><th>Dataset</th><th>Contents</th><th>Source</th><th>Benchmarks</th><th>Notes</th></tr>
	<tr><td><a href="http://www.cvlibs.net/datasets/kitti/">KITTI</a></td><td>3D and bird's eye view object detection, Depth completion and prediction, Semantic segmentation + more</td><td>KIT, TTI-C</td><td></td><td></td></tr>
	<tr><td><a href="https://arxiv.org/abs/1612.00423">TorontoCity Benchmark</a></td><td>712.5km2 of land, 8k km of road, 400k buildings</td><td>University of Toronto(, Uber?)</td><td></td><td>Aimed to be released in 2018</td></tr>
	<tr><td><a href="http://bair.berkeley.edu/blog/2018/05/30/bdd/">BDD100K</a></td><td>120M images spread across 100k videos, covers different weather conditions. Richly annotated (objects, road lines, drivable areas). Subset of 10k images with full-frame instance segmentation</td><td>UC Berkeley's DeepDrive with big industrial partners e.g. Honda, Toyota, Ford</td><td></td><td>Jack Clark opinion: 'multi-modal dataset, could be used to evaluate transfer learning from other systems'</td></tr>
	<tr><td><a href="https://arxiv.org/abs/1805.07193">EuroCity Persons</a></td><td>for object and pedestrian detection, 45k hand-annotated images comprising over 100k pedestrians in different weather settings</td><td></td><td></td><td>**diversity** that could help with generalisation: 31 cities in 12 European countries</td></tr>
	<tr><td><a href="https://www.cityscapes-dataset.com/">Cityscapes</a></td><td>5k high qual, 20k coarse annotated images with 30 classes, 50 different cities</td><td>Daimler, Max Planck Institute, TU Darmstadt</td><td></td><td>Has sub-datasets like CityPersons (person annotations on top of CityScapes)</td></tr>
	<!-- <tr><td>title</td><td>contents</td><td>source</td><td>benchmarks</td><td>notes</td></tr> -->	
</table>

## Images
- CIFAR-10
	Example image classification dataset: CIFAR-10. One popular toy image classification dataset is the CIFAR-10 dataset. This dataset consists of 60,000 tiny images that are 32 pixels high and wide. Each image is labeled with one of 10 classes (for example “airplane, automobile, bird, etc”). These 60,000 images are partitioned into a training set of 50,000 images and a test set of 10,000 images. In the image below you can see 10 random example images from each one of the 10 classes. (Source: CS231n)
- [Satellite imagery: DeepGlobe](https://arxiv.org/abs/1805.06561)
	- Road extraction, building detection, land cover classification


## Video
- [Moments in Time (MIT)](https://arxiv.org/abs/1801.03150)
	- 1M videos, each of which is 3s long
	- Initial labels: 339 verbs linked to a variety of different actions or activities
	- 'Future versions of the dataset will include multi-labels action description (i.e. more than one action occurs in most 3-second videos), focus on growing the diversity of agents, and adding temporal transitions between the actions that agents performed'

## Action Classification and Localisation
- SLAC
	- 200 action classes, 520k videos, over 1.75M individual annotations
	- Labelled using AI systems and human feedback (/humans for clips AI system can't label with high confidence)

## Medical
- [Standardised nail dataset (Han and Park et. al., Jan 2018)](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191493)
	- c. 50k nails

## Obtaining or Improving datasets
- [Using noisy crowd annotations from non-experts]
	- [[Paper: Adversarial Learning for Chinese NER from Crowd Annotations (Yang et. al., Jan 2018)]](https://arxiv.org/abs/1801.05147)
	- Model: uses common Bi-LSTM and private Li-STM for representing annotator-generic and annotator-specific information.
	- Created two datasets for Chinese Named Entity Recognition (NER) tasks