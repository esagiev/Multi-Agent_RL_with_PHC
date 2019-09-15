#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Developed by Erkin Sagiev, esagiev.github.io
# Version on 15 September 2019.

import numpy as np # To avoid "np is not defined" error.

class matrix_game():
    agent_num = 2
    
    def agent_set(self):
        """ Returns the set of agents. """
        return np.arange(self.agent_num)
    
    def action_set(self):
        """ Returns the set of actions. """
        return np.arange(self.paymat.shape[0])
    
    def payoff(self):
        """ Returns payoff matrices. """
        return np.asarray([self.paymat, self.paymat])
    
    def best_resp(self, actions):
        """ Returns both agents' best response to another agent's action. """
        try:
            return np.argmax(
                      self.payoff()[self.agent_set(), :, np.flip(actions)], 1
                      )
        except:
            return print('Wrong input. Array with values in {0,1}.')


class wrap(matrix_game):
    """ Two agent game created from given data. Takes two payoff matrices. """
    def __init__(self, paymat_1, paymat_2):
        self.paymat_1 = paymat_1
        self.paymat_2 = paymat_2
    
    def payoff(self):
        """ Returns payoff matrices. """
        return np.asarray([self.paymat_1, self.paymat_2])


class matching_pennies(matrix_game):
    """ Matching Pennies game with two agents.
    Actions are Head and Tail.
    """
    paymat = np.array([[1, -1],[-1, 1]])
    
    def payoff(self):
        """ Returns payoff matrices. """
        return np.asarray([self.paymat, -self.paymat])


class prisoner_dilemma(matrix_game):
    """ Prisoner's dilemma game with two agents.
    Actions are Defect and Confess.
    """
    paymat = np.array([[-1, -3],[0, -2]])


class stag_hunt(matrix_game):
    """ Stag hunt game with two agents.
    Actions are Stag or Hare.
    """
    paymat = np.array([[3, 0],[2, 1]])


class RPS(matrix_game):
    """ Rock-Paper-Scissors game.
    Actions are Rock, Paper, and Scissors.
    """
    paymat = np.array([[0, -1, 1],[1, 0, -1],[-1, 1, 0]])


class hawk_dove(matrix_game):
    """ Hawk and dove game.
    Actions are Dare (Hawk) and Chicken (Dove).
    """
    paymat = np.array([[0, -1],[1, -2]])

