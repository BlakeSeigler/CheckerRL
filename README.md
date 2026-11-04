Model Based Planning
MCTS + NN (like alphaGO)
 
# Strategies
Checkers is a game with many possible combinations. Too many to absolutely compute values. Thus for strategies I'll try out a few different strategies. 

## Monte Carlo Tree Search (MCTS) 
Very common algorithm for optimization approaches where the sample space is incredibly large (like 1e171 large)
Is a heuristic search algorithm, not inherently RL but can be made that way if we use an NN to represent the policy and value determination. Apparently this is sort of an actor critic method.


TODO
- finish running a clean run to ensure that nothing breaks
- build a gpu server to run multiple training games in parallel for training
- build a testing rig with prior checkpoints