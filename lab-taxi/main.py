from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent )
#avg_rewards, best_avg_reward = interact(env, agent, 'SARSA_MAX')
#avg_rewards, best_avg_reward = interact(env, agent, 'EXPECTED_SARSA')
