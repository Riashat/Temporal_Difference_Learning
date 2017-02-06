import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random


from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from collections import defaultdict
from lib.envs.gridworld import GridworldEnv
from lib import plotting

env = GridworldEnv()

matplotlib.style.use('ggplot')



def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn




def double_q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q_A = defaultdict(lambda : np.zeros(env.action_space.n))

	Q_B = defaultdict(lambda : np.zeros(env.action_space.n))

	Total_Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	# state = 0
	# actions_init = 0
	# Total_Q[state][actions_init] = Q_A[state][actions_init] + Q_B[state][actions_init]

	#choose a based on Q_A for now
	policy = make_epsilon_greedy_policy(Total_Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		state = env.reset()

		for t in itertools.count():

			#choose a from policy derived from Q1 + Q2 (epsilon greedy here)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)			
			# with taken aciton, observe the reward and the next state
			next_state, reward, done, _, = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#choose randomly either update A or update B
			#randmly generate a for being 1 or 2
			random_number = random.randint(1,2)

			if random_number == 1:
				best_action_Q_A = np.argmax(Q_A[next_state])
				TD_Target_A = reward + discount_factor * Q_B[next_state][best_action_Q_A]
				TD_Delta_A = TD_Target_A - Q_A[state][action]
				Q_A[state][action] += alpha * TD_Delta_A

			elif random_number ==2:
				best_action_Q_B = np.argmax(Q_B[next_state])
				TD_Target_B = reward + discount_factor * Q_A[next_state][best_action_Q_B]
				TD_Delta_B = TD_Target_B - Q_B[state][action]
				Q_B[state][action] += alpha * TD_Delta_B


			if done:
				break

			state = next_state
			Total_Q[state][action] = Q_A[state][action] + Q_B[state][action]


	return Total_Q, stats



def main():
	Total_Q, stats= double_q_learning(env, 1000)
	plotting.plot_episode_stats(stats, 200)



if __name__ == '__main__':
	main()





