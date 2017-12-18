# Environments

<!--TODO: add info on OpenAI Gym, DeepMind Lab,  Malmo-->
<!--TODO: categorise wrappers like OpenAI Gym vs envs like HoME -->

* OpenAI Gym
* DeepMind Lab
* Atari/Arcade Learning Environment (ALE)
    * accessible via OpenAI Gym interface
    * [updated with modes and difficulties in 2017](http://www.marcgbellemare.info/introducing-the-ale-6/)
    * [[code]](https://github.com/mgbellemare/Arcade-Learning-Environment)

<!--TODO: elab on what Gridworlds is -->
* Gridworlds (for AI Safety) 
	- [[code: pycolab]](https://github.com/deepmind/pycolab)
	- [[Paper]](https://arxiv.org/abs/1711.09883)

* Malmo (Microsoft, Minecraft-based)
* Unity Machine Learning Agents
	- [support for curriculum learning in v0.2](https://blogs.unity3d.com/2017/12/08/introducing-ml-agents-v0-2-curriculum-learning-new-environments-and-more/)

## 3D Home Environments
* HoME (Household Multimodal Environment, 45k 3D houses populated with objects)
	- Supports sound, vison and touch sensing
	- [[Website]](https://home-platform.github.io/) [[Code]](https://github.com/HoME-Platform/home-platform) [[Paper]](https://arxiv.org/abs/1711.11017v1), 
	- OpenAI Gym-compatible
- AI2-THOR: 3D agent-training environment (The House Of inteRactions (Dec 2017)
	- 120 actionable, high quality 'photo-realistic' 3D scenes (rooms in a house)
		- Actions like: pick up mug, put mug in coffee  machine, open fridge.
		- Claim: realistic, interactive scenes mean better transfer of learned models to the real world
	- Scenes are hand-crafted
	- Unity-based with Python API. OS: Mac or Ubuntu.
	- [[Website]](http://ai2thor.allenai.org./)[[Video]](https://www.youtube.com/watch?time_continue=7&v=MvvAhF4HZ8s) [[Paper]](https://arxiv.org/abs/1712.05474)