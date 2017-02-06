#implementation of the SARSA algorithm:

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys



from collections import defaultdict
from lib.envs.gridworld import GridworldEnv
from lib import plotting
env = GridworldEnv()


# from collections import defaultdict
# from lib.envs.cliff_walking import CliffWalkingEnv
# from lib import plotting
# env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):

	#epsilon greedy policy based on a given Q function and epsilon
	# Q is a dictionary - maps each state to a action values - where each value is a numpy array of length nA

	#this will return a function itself
	#the function returns the probabilities of taking each action

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn

def chosen_action(Q):
	best_action = np.argmax(Q)
	return best_action



def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	action_space = env.action_space.n

	#on-policy which the agent follows - we want to optimize this policy function
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	#each step in the episode
	for i_episode in range(num_episodes):

		state = env.reset()
		action_probs = policy(state)

		#choose a from policy derived from Q (which is epsilon-greedy)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		#for every one step in the environment
		for t in itertools.count():
			#take a step in the environment
			# take action a, observe r and the next state
			next_state, reward, _, _ = env.step(action)
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(action_probs)), p = next_action_probs)


			next_2_state, reward_2_step, done, _ = env.step(next_action)
			next_2_action_probs = policy(next_2_state)
			next_2_action = np.random.choice(np.arange(len(next_2_action_probs)), p = next_2_action_probs)


			# stats.episode_rewards[i_episode] += reward
			# stats.episode_lengths[i_episode] = t

			stats.episode_rewards[i_episode] += reward_2_step
			stats.episode_lengths[i_episode] = t


			# TD Update Equations
			#TD Target - One step ahead
			td_target = reward + discount_factor * reward_2_step + discount_factor*discount_factor * Q[next_2_state][next_2_action]
			
			# TD Error
			td_delta = td_target - Q[state][action]

			Q[state][action] += alpha * td_delta

			
			if done:
				break
			action = next_action
			state = next_state


	return Q, stats




def main():
	Q, stats= sarsa(env, 500)
	plotting.plot_episode_stats(stats)



if __name__ == '__main__':
	main()
