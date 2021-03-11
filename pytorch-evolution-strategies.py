import gym
import numpy as np
import torch
from collections import deque


INPUTS = 4
HIDDEN = 8
OUTPUTS = 2
TARGET = 190
POPULATION = 5

ave_reward = deque(maxlen=100)
short_ma = deque(maxlen=5)
long_ma = deque(maxlen=10)

env = gym.make('CartPole-v0')
env.seed(9); torch.manual_seed(1); np.random.seed(1)


class EvolutionStrategies(torch.nn.Module):
	def __init__(self, inputs, hidden, outputs, target, population):
		super(EvolutionStrategies, self).__init__()
		self.linear1 = torch.nn.Linear(inputs, hidden)
		self.linear2 = torch.nn.Linear(hidden, outputs)
		self.population_size = population
		self.sigma = 0.1
		self.learning_rate = 0.0001
		self.counter = 0
		self.rewards = []
		self.score_tracking = deque(maxlen = 100)
		self.master_weights = []
		self.target = target

		for param in self.parameters():
			self.master_weights.append(param.data)
		self.populate()

	def forward(self, x):
		x = torch.relu(self.linear1(x))
		return self.linear2(x)

	def populate(self):
		self.population = []
		for _ in range(self.population_size):
			x = []
			for param in self.parameters():
				x.append(np.random.randn(*param.data.size()))
			self.population.append(x)

	def add_noise_to_weights(self):
		for i, param in enumerate(self.parameters()):
			noise = torch.from_numpy(self.sigma * self.population[self.counter][i]).float()
			param.data = self.master_weights[i] + noise
		self.counter += 1

	def log_reward(self, reward):
		# When we've got enough rewards, evolve the network and repopulate
		self.rewards.append(reward)
		if len(self.rewards) >= self.population_size:
			self.counter = 0
			self.evolve()
			self.populate()
			self.rewards = []
		self.add_noise_to_weights()

	def evolve(self):
		# Multiply jittered weights by normalised rewards and apply to network
		if np.std(self.rewards) != 0:
			normalized_rewards = (self.rewards - np.mean(self.rewards)) / np.std(self.rewards)
			for index, param in enumerate(self.parameters()):
				A = np.array([individual[index] for individual in self.population])
				rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T).float()
				param.data = self.master_weights[index] + self.learning_rate / (self.population_size * self.sigma) * rewards_pop
				self.master_weights[index] = param.data

		# Adaptive learning rate (work in progress)
		high_score = np.max(self.rewards)
		self.score_tracking.append(high_score)
		self.learning_rate = (self.learning_rate*5 + (self.target - np.mean(self.score_tracking))*0.000005)/6
		self.sigma = self.learning_rate * 10


model = EvolutionStrategies(INPUTS, HIDDEN, OUTPUTS, TARGET, POPULATION)
state = env.reset()
steps = 200
episodes = 30000


for episode in range(episodes):
	episode_reward = 0	
	state = env.reset()

	for s in range(steps):
		# env.render()
		action = torch.argmax(model.forward(torch.FloatTensor(state)))
		state, reward, done, _ = env.step(int(action))
		episode_reward += reward
		if done:
			model.log_reward(episode_reward)
			break

	ave_reward.append(episode_reward)
	short_ma.append(episode_reward)
	long_ma.append(episode_reward)


	if episode % 10 == 0:
		print(episode, 'Average reward: ', np.mean(ave_reward), ' LR: ', model.learning_rate)
		if np.mean(short_ma) < np.mean(long_ma):
			model = EvolutionStrategies(INPUTS, HIDDEN, OUTPUTS, TARGET, POPULATION)
			short_ma.clear()
			long_ma.clear()


	if np.mean(ave_reward) >= env.spec.reward_threshold:
		print('Completed at episode: ', episode)
		break


