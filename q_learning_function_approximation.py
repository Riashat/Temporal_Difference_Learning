import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing


from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make("MountainCar-v0")
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action



observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])

featurizer.fit(scaler.transform(observation_examples))




#Class for Value Function Approximator:
class Estimator():

	def __init__(self):
		self.models = []
		for _ in range(env.action_space.n):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)


	#function to return the featurized representation of a state
	def featurize_state(self, state):
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]

	# function to define the value function predictions
	def predict(self, s, a=None):

		features = self.featurize_state(s)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]

	def update(self, s, a, y):
		features = self.featurize_state(s)
		self.models[a].partial_fit([features], [y])






def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn




def q_learning(env, estimator, num_episodes, discount_factor = 1.0, epsilon = 0.1, epsilon_decay=1.0):

	#implementation of Q learning for off-policy TD control with function approximation
	#off-policy - epsilon greedy policy
	# find optimal greedy policy
	    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):

    	# greedy policy that the agent is following
    	policy = make_epsilon_greedy_policy(estimator, epsilon*epsilon_decay**i_episode, env.action_space.n)


    	#reset environment
    	state = env.reset()


    	for t in itertools.count():

    		#choose action based on policy derived from Q
    		action_probs = policy(state)
    		action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

    		#take step in the environment
    		next_state, reward, done, _ = env.step(action)

    		#update cumulative rewards
    		stats.episode_rewards[i_episode] += reward
    		stats.episode_lengths[i_episode] = t


    		#TD Update
    		q_values_next = estimator.predict(next_state)

    		#Q-value TD Target
    		td_target = reward + discount_factor* np.max(q_values_next)

    		#update function approximator using the target value
    		estimator.update(state, action, td_target)


    		if done:
    			break

    		state = next_state

    return stats, action



def main():

	estimator = Estimator()
	stats, action= q_learning(env, estimator, 100, epsilon=0.0)

	for _ in range(1000):
		env.reset()
		env.render()
		env.step(action)

	# plotting.plot_episode_stats(stats)



if __name__ == '__main__':
	main()



