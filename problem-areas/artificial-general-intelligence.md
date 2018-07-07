# Artificial General Intelligence (AGI)

There are currently a few popular approaches to AGI:

### Game-playing
- AlphaGo: Deep RL
- Libratus (Poker, partially observable environment): Non-deep RL
- Atari games: often deep RL
	- Some approaches incorporate more prior knowledge, e.g. Vicarious
- DoTA 2, King of Glory (Tencent)
	- OpenAI DoTA bot (1v1)
		- still use human-chosen inputs that may be key to success
		- + other caveats
	- [OpenAI Five: DoTA team of bots (5v5 subject to certain restrictions)](https://blog.openai.com/openai-five/)
		- Beat amateur teams and beat certain semi-pros and pros in some games (June 2018). Show signs of long-term planning, learned fairly advanced strategies from only self-play
		- Surprising since it is just PPO, would
		- Another key was further annealing the exponential decay factor so the half-life was 5min (vs usually under 1s to a few seconds)
- IBM Watson x Jeopardy sort (sort of game-playing)


### Language
- FAIR had some open competition on this


### others
- Visual semantic planning or sth may qualify as beginnings of 'intelligence'
- Meta learning