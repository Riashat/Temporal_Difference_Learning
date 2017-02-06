import gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random


from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()



def make_epsilon_greedy_policy(Q, epsilon, nA):

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
			next_state, reward, done, _ = env.step(action)

			#choose a' from s' using policy derived from Q
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(action_probs)), p = next_action_probs)

			#update cumulative count of rewards based on action take (not next_action) using Q (epsilon-greedy)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# TD Update Equations
			#TD Target - One step ahead
			td_target = reward + discount_factor * Q[next_state][next_action]
			
			# TD Error
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta
			if done:
				break
			action = next_action
			state = next_state


	return Q, stats



def sarsa_2_step_TD(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


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





def q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		state = env.reset()

		for t in itertools.count():

			#take a step in the environmnet
			#choose action A using policy derived from Q
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			
			# with taken aciton, observe the reward and the next state
			next_state, reward, done, _, = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			# TD Update Equations:

			# max_a of Q(s', a) - where s' is the next state, and we consider all maximising over actions which was derived 
			#from previous policy based on Q
			best_next_action = np.argmax(Q[next_state])

			td_target = reward + discount_factor * Q[next_state][best_next_action]

			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state


	return Q, stats




def q_learning_2_step_TD(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):



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

			td_target = reward +  discount_factor * reward_2_step +   discount_factor * discount_factor * Q[next_2_state][best_next_action]

			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state


	return Q, stats






def double_q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q_A = defaultdict(lambda : np.zeros(env.action_space.n))

	Q_B = defaultdict(lambda : np.zeros(env.action_space.n))

	Total_Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#choose a based on Q_A + Q_B
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








def plot_episode_stats(stats1, stats2, stats3, stats4, stats5, smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="Sarsa")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Q Learning")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Sarsa 2 Step TD")
    cum_rwd_4, = plt.plot(rewards_smoothed_4, label="Q Learning 2 Step TD")
    cum_rwd_5, = plt.plot(rewards_smoothed_5, label="Double Q Learning")

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()


    return fig



def main():

	Number_Episodes = 1500
	print "SARSA"
	Sarsa_Q, stats_sarsa= sarsa(env, Number_Episodes)
	
	print "Q-Learning"
	Q_learning_Q, stats_q_learning= q_learning(env, Number_Episodes)
	
	print "2 step SARSA"
	Sarsa_Q_2_step_TD, stats_sarsa_2_step = sarsa_2_step_TD(env, Number_Episodes)

	print "2 step Q-Learning"
	Q_learning_Q_2_step_TD, stats_q_learning_2_step = q_learning_2_step_TD(env, Number_Episodes)

	print "Double Q Learning"
	Doube_Q, stats_Double_Q = double_q_learning(env, Number_Episodes)



	plot_episode_stats(stats_sarsa, stats_q_learning, stats_sarsa_2_step, stats_q_learning_2_step, stats_Double_Q)



if __name__ == '__main__':
	main()
