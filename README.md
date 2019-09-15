15 September 2019

The scripts were developed as a review of existing reinforcement learning tools.

About games.py

It has 6 subclasses:

1) matching_pennies
2) prisoner_dilemma
3) stag_hunt
4) RPS (Rock, Paper, and Scissors)
5) hawk_dove
6) wrap

The last class takes two numpy 2 dimensional arrays and returns a game with
the arrays as payoff matrices.

All games have only 2 agents with equal number of actions.

All classes have 4 methods:

1) agent_set - Returns set of agents.
2) action_set - Returns set of actions.
3) payoff - Returns payoff matrices of both agents.
4) best_resp - Takes array of actions and returns the best responses.


About qlearn.py

It has only one function - PHC (Policy Hill Climbing).
It includes both PHC and WoLF PHC (Win or Learn Fast).

The algorithms were taken from:
    Win or Learn Fast Policy Hill-Climbing algrorithm
    is taken from Bowling, M. and Veloso, M., 2002.
    Multiagent learning using a variable learning rate.
    Artificial Intelligence, 136(2), pp.215-250.

As arguments it takes:

1) game
2) immediate reward
3) gamma (discount parameter)
4) alpha_param (share of maximum Q in Q-value)
   Takes [b, a] and puts into alpha = 1/(b + a*t).
   Variable t is current itiration.
5) delta_param (change of policy)

   If WoLF, takes 3 values [b, a, c] and
   puts into delta for win strategy: 1/(b + a*t)
   and into delta for loss strategy: c*(1/(b + a*t))
   
   If not-WoLF, takes 1 value as a fixed delta in [0,1].

6) iter_max (maximum number of iterations)
7) explore_rate (exploration rate in [0,1])
8) WoLF (turns on WoLF PHC when True)


Jupyter notebook example.ipynb includes a demonstrative execution code.

