# the MCTS agents for Blood bowl
## Prerequisites
We need use the botbowl framework that has described in assignment.

## Start the MCTS Search
In the process of finding a solution, we completed two different implementations of MCTS: <br>

* One is to expand one child node at a time, using the following command to start and the default computation budget=100:

```
python MCTS1/main.py

```

* The other is to expand all the children nodes of the selected node at once, using the following command to start and the default playout = 7:

```
python MCTS2/MCTSBot.py
```

