import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys



from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')


env = CliffWalkingEnv()




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




def q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		state = env.reset()

		for t in itertools.count():

			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			next_state, reward, done, _, = env.step(action)


			next_2_state, reward_2_step, done, _ = env.step(action)
			next_2_action_probs = policy(next_2_state)
			next_2_action = np.random.choice(np.arange(len(next_2_action_probs)), p = next_2_action_probs)


			# stats.episode_rewards[i_episode] += reward
			# stats.episode_lengths[i_episode] = t

			stats.episode_rewards[i_episode] += reward_2_step
			stats.episode_lengths[i_episode] = t

			# TD Update Equations:


			best_next_action = np.argmax(Q[next_2_state])

			td_target = reward +  discount_factor * reward_2_step +   discount_factor * discount_factor * Q[next_state][best_next_action]

			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state


	return Q, stats


def main():
	Q, stats= q_learning(env, 300)
	plotting.plot_episode_stats(stats)



if __name__ == '__main__':
	main()





