# Pytorch Evolution Strategies
A very simple implementation of Evolution Strategies with Pytorch and CartPole.

## Updated to solve CartPole quickly
In its current form, this implementation of Evolution Strategies solves CartPole after 6 episodes (getting a perfect run by 106).

As always, it depends on which environment seed is used – ensuring it gets the right start. This seems to be a big problem with most OpenAI challenges. However, this implementation works quite well across environments. But as a temporary 'fix' to help it generalise, I've added a Pestilence function:
– Short_ma (moving average) and Long_ma track episode rewards, and if the trend is unfavourable (i.e. it's not converging on a solution) it kills the population off and starts again.

Getting a good start has always been more important than I'd like with these challenges, so wiping the old population out is often much quicker than persisting down evolutionary blackholes.

What's most promising about ES to me is not needing back-propagation. Presumably, you could stuff these models with any functions you want, and still converge on solutions. However, the issue with it getting stuck on bad solutions (that necessitates Pestilence) is not ideal, and something I'm sure could be better ironed out.

Also, the sensitivity of things like the adaptive learning rate, in finding solutions at all, feels very ad-hoc. I'm puzzled why this approach isn't more robust and doesn't work better (despite getting some pretty decent scores here).

However, I think there's limitless opportunity to modify how this system works and fix these problems. There are many lessons from genetics that can be applied directly here. I'd hope to get this system off the messy solutions I'm using here (or at least make them more elegant), and then start applying it to much more complex problems and experimental neural net functions.

## How to use
Initialise your model with the number of inputs and outputs you want, and the reward target for the particular task (this just helps with the adaptive learning rate, which is a work in progress):

**model = EvolutionStrategies(inputs=4, outputs=2, target=190)**

Get the recommended action based on the current state:

**torch.argmax(model.forward(torch.FloatTensor(state)))**

Update your model with its reward:

**model.log_reward(episode_reward)**

That's it.

## How it works
1. Initialise a simple neural network;
2. Store the network's weights as master_weights;
3. Create a population of 15 different sets of noise (random weights);
4. Run the CartPole simulation with the first set of noise applied to the weights, and once it's done, send the reward back to the network;
5. The network logs the reward, resets the weights, then applies the next stored noise to the weights;
6. Once the network's logged 15 different sets of rewards, it evolves the master_network and goes back to step 3.

The evolving bit's just a case of multiplying our population (of 15 sets of noisey weights) by the (normalised) rewards they achieved. So the best result of our 15 is strongly imprinted on the master network ("Do more of this."), and the worst result is negatively imprinted ("Do less of that.")

I've only just started experimenting with ES, so this is a first attempt to design a 'toy code' implementation I find easy to work with.

This solves CartPole by episode 231. Not the fastest or most robust (yet). But fairly easy to apply to a variety of tasks.

