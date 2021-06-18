from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
<<<<<<< HEAD
avg_rewards, best_avg_reward = interact(env, agent )
#avg_rewards, best_avg_reward = interact(env, agent, 'SARSA_MAX')
#avg_rewards, best_avg_reward = interact(env, agent, 'EXPECTED_SARSA')
=======
avg_rewards, best_avg_reward = interact(env, agent)
>>>>>>> 8f30861f8dd600672a58df9907d3b74f8999f1dd
