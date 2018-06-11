# Datasets

Note: This page is currently more a list than an indicator of which are the 'main datasets' in research.

## Speech
- [Common Voice (Mozilla)](https://voice.mozilla.org/data)
	- 400 hours of speech from 400k recordings from 20k people.

## Robotics
- [RoboCupSimData](https://bitbucket.org/oliverobst/robocupsimdata/)
	- Over 180 hours of robot soccer gameplay.
	- [[Paper]](https://arxiv.org/abs/1711.01703) [[Data]](https://bitbucket.org/oliverobst/robocupsimdata/)

## Self-Driving Cars
- KITTI
	- New benchmarks (Feb-Mar 2018): 3D object detection, Bird’s eye view object detection, Depth completion, Single image depth prediction, Semantic segmentation, Semantic instance segmentation.
	- Robust Vision Challenge 2018 at CVPR 
- [TorontoCity Benchmark](https://arxiv.org/abs/1612.00423) (from U of Toronto, aimed to be released in 2018): 
	- 712.5km2 of land, 8k km of road, 400k buildings
- BDD100K (UC Berkeley's DeepDrive (partners with Honda, Toyota, Ford), Nexar)
	- 120M images spread across 100k videos, covers different weather conditions. Richly annotated (objects, road lines, drivable areas). Subset of 10k images with full-frame instance segmentation
	- (Jack Clark opinion: 'multi-modal dataset, could be used to evaluate transfer learning from other systems')
- [EuroCity](https://arxiv.org/abs/1805.07193)
	- **diversity** that may help with generalisation: 31 cities in 12 European countries
	- for object and pedestrian detection, 45k images comprising over 100k pedestrians in different weather settings
	- pedestrians and vehicle-riders are hand-annotated, confounding images (posters of people, reflections in windows) also annotated
- CityPersons

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