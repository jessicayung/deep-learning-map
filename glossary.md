# Glossary

(Work in progress. The focus is on terms that won't be part of most introductory courses since those definitions are easy to find and are usually in the [WildML glossary](http://www.wildml.com/deep-learning-glossary/).)


### Frameworks


#### Options framework
- involves abstractions over the space of actions
- at each step, the agent chooses either a one-step 'primitive' action or a 'multi-step' action policy (option). Each option defines a policy over actions (either primitive or other options) and can be terminated according to a stochastic function of $\beta$.
- Paper: Sutton et. al.

Definition from Kulkarni and Narasimhan et. al (2016) 

- Policy gradient methods

### Training methods

- Backpropagation
- Synthetic gradients

### Models

- A3C (Asynchoronous Advantage Actor-Critic)
- A2C (Synchronous A3C: Advantage Actor-Critic)
- Rainbow
- Neural Turing Machine
- DQN
- Capsule Networks
- Dilated convolutions
- Skip connections
- Dilated LSTMs