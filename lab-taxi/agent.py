import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6 ):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha=0.1
        self.eps = 1.0
        self.eps_min = 0.005
        self.episode_i = 1
        self.method = 'EXPECTED_SARSA'

    def select_action(self, state):
        """ Given the state, select an action.

        It implements an e-greedy policy

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        #Implementing an eps-greedy

        if state in self.Q and random.uniform(0,1) > self.eps:
            return np.argmax( self.Q[state] )
        else:
            return np.random.choice(self.nA)


    def update_q_sarsamax( self, state, action, reward, next_state ):
        q_current = self.Q[state][action]
        q_target = np.max( self.Q[next_state] ) + reward
        return q_current + self.alpha*( q_target - q_current )

    def update_q_sarsa( self, state, action, reward, next_state ):
        q_current = self.Q[state][action]
        next_expected_action = self.select_action( state )
        q_target = self.Q[next_state][next_expected_action] + reward
        return q_current + self.alpha*( q_target - q_current )        


    def update_q_expected_sarsa( self, state, action, reward, next_state ):
        q_current = self.Q[state][action]

        p_probs = np.ones( self.nA )*(self.eps)/self.nA
        p_probs[  np.argmax( self.Q[ next_state ] ) ] = 1.0 - ( self.nA - 1 )*self.eps/self.nA


        expected_Q = np.dot( self.Q[next_state], p_probs )
        q_target = expected_Q + reward

        return q_current + self.alpha*( q_target - q_current )


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if( self.method == 'SARSA' ):
            self.Q[state][action] = self.update_q_sarsa( state, action, reward, next_state )

        elif( self.method == 'SARSA_MAX' ):
            self.Q[state][action] = self.update_q_sarsamax( state, action, reward, next_state )

        elif( self.method == 'EXPECTED_SARSA' ):
            self.Q[state][action] = self.update_q_expected_sarsa( state, action, reward, next_state )

        if done:
            self.eps = max( 1.0/self.episode_i, self.eps_min ) #Deixa 1% de probabilidade de explorar
            self.episode_i += 1    