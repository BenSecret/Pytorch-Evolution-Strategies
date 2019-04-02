# Pytorch Evolution Strategies
A very simple implementation of Evolution Strategies with Pytorch and CartPole.

## How to use
Initialise your model with the number of inputs and outputs you want, and the reward target for the particular task (this just helps with the adaptive learning rate, which is a work in progress).

model = EvolutionStrategies(inputs=4, outputs=2, target=190)

Update your model with its reward.

model.log_reward(episode_reward)

That's it.

## How it works
1. Initialise a simple neural network;
2. Store the network's weights as master_weights;
3. Create a population of 15 different sets of noise (random numbers) we can apply directly to our network's weights;
4. Run the CartPole simulation as normal, and once it's done, send the reward back to our network;
5. The network logs the reward, resets the weights, then applies noise to the weights from one of our population;
6. Once the network's logged 15 different rewards, it evolves the master_network and goes back to step 3.

The evolving bit's just a case of multiplying our population (of 15 sets of noise) by the (normalised) rewards they achieved. So the best result of our 15 is strongly imprinted on the master network ("Do more of this."), and the worst result is negatively imprinted ("Do less of that.")

I've only just started experimenting with ES, so this is a first attempt to design an implementation I find easy to work with.

This solves CartPole by episode 231. Not the fastest or most robust (yet). But very easy to apply to a wide variety of tasks.

