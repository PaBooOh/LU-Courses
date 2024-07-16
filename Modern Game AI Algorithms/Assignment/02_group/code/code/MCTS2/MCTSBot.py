#!/usr/bin/env python3

from collections import deque
import botbowl
import numpy as np
from copy import deepcopy
from MCTS import MCTSNode

# set (hyper)parameters
MCTS_PLAYOUT_NUM = 7

class MCTS():
    """
    :param
    root: the root node corresponding to the game state that need to search for the best action
    root_state: the current state id, for revert
    team: represents which team will perform an action in this state (step).
    game: current game env
    """
    def __init__(self, game, current_team, historical_real_action_sequence):
        self.root = MCTSNode(state_steps = None, team = current_team) # parent and action are None. State is integer id (for revert) here rather than steps (for forward)
        self.root_state = game.get_step()
        self.team = current_team
        self.game = game
        self.historical_real_action_sequence = historical_real_action_sequence

    """
    Do 1 time MCTS playout where playout including select, expand, simulate and backup.
    """
    def playout(self):
        # Initialization
        node = self.root # initialize the root node first.
        print('---- Which would be the real action to be chosen: ')
        print(self.game.state.available_actions)
        print()
        while True: # (1) Selection
            if node.isLeaf():
                break
            node = node.select()
            self.game.forward(node.state) # get to the state (from current to future) of the leaf node selected.
            print('>>Path selected using UCB: ', node.action.action_type, node.action.player, node.action.position)
        
        if not self.game.state.game_over: # if the leaf node selected with UCB does not represents game over (leaf node in mid of tree)
            leaf_node_state_id = self.game.get_step() # For revert
            node.expand(self.game, self.root, self.historical_real_action_sequence) # (2) Expansion: check if the current state is the terminal before expand all actions at a time.
            for child in node.getChildren(): # (3) Simulation: simulate all expanded nodes that have yet to be simulated
                if child.getVisitNum() == 0:
                    child.simulate(self.game)
                    self.game.revert(leaf_node_state_id) # revert to the state of the leaf node selected for following simulation
                    
        else: # if the leaf node selected with UCB represents game over, simulation is omitted and leap to apply backup.
            print('Selection pharse: Terminal leaf node was selected')
            winner_team = self.game.get_winning_team()
            node.backpropagate(winner_team)
        
        # self.game = self.root_game # revert to the initial state before the next playout
        self.game.revert(self.root_state) # revert to the initial state before the next playout

    """
    Get the action corresponding to the most visited node
    """
    def getMostVistedAction(self):
        child_with_most_visited = max(self.root.children, key=lambda child: child.getVisitNum())
        action_with_most_visited = child_with_most_visited.getAction()
        return action_with_most_visited
        

"""
MCTS Agent
"""
class MCTSBot(botbowl.Agent):

    """
    :param
    playout_num: the number of playout(select, expand, simulate and backup)
    rnd: random seed
    formation_setup: Is the formation set
    historical_real_action_sequence: previous three real actions that the agent performs in the game is recorded
    """
    def __init__(self, name, seed=None):
        super().__init__(name)
        self.playout_num = MCTS_PLAYOUT_NUM
        self.rnd = np.random.RandomState(seed)
        self.formation_setup = False
        self.historical_real_action_sequence = deque(maxlen=3)

    def new_game(self, game, team):
        self.my_team = team

    """
    Search for the best action in the current game state
    """
    def getBestAction(self, game):
        # Rebuild the tree for every decision-making
        # Config
        game_copy = deepcopy(game) # search but not change the real game.
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True

        # Apply MCTS for planning
        mcts = MCTS(game_copy, self.my_team, self.historical_real_action_sequence) # need to take as input the current team planning the next action using MCTS.
        for i in range(self.playout_num):
            print('========> Playout ', i+1, ' start <========')
            print('Historical actions: ', [action.action_type for action in self.historical_real_action_sequence])
            mcts.playout()
            print()
            print()

        # After a certain amount of playouts, choose the reasonable action (here is to choose the action with most visited time, as the real action).
        action = mcts.getMostVistedAction()
        self.historical_real_action_sequence.append(action)
        print('******** Real action: ', action.action_type)
        return action

    """
    is called at every step in the game where the agent is supposed to perform an action.
    """
    def act(self, game):
        # Formation and kick/receive setup
        # Explictly determine heads/tails, kick/receive and formations before MCTS in order to reduce size of tree
        action_choices = game.state.available_actions
        actions_type = [action_choice.action_type for action_choice in action_choices]
        # print(actions_type)
        # (1) determine heads or tails then kick or receive
        if botbowl.ActionType.HEADS in actions_type or botbowl.ActionType.RECEIVE in actions_type:
            action_choice = self.rnd.choice(game.state.available_actions)
            action = botbowl.Action(action_choice.action_type)
            self.historical_real_action_sequence.append(action)
            return action
        
        # (2) determine formations without considering custom
        # (2.1) formation
        if botbowl.ActionType.PLACE_PLAYER in actions_type and self.formation_setup == False:
            action_choice = self.rnd.choice(game.state.available_actions[2:]) # randomly choose a foramtion from the last two/one index
            self.formation_setup = True # formations is selected but have yet to end setup
            action = botbowl.Action(action_choice.action_type)
            self.historical_real_action_sequence.append(action)
            return action
        # (2.2) end_setup
        if botbowl.ActionType.PLACE_PLAYER in actions_type and self.formation_setup == True:
            action_choice = game.state.available_actions[1] # END_SETUP
            self.formation_setup = False
            action = botbowl.Action(action_choice.action_type)
            self.historical_real_action_sequence.append(action)
            return action
        # (3) Reduce redundant action leading to small size of tree. (E.g., choose END_TURN if there are no other actions)
        if len(actions_type) == 1:
            if len(action_choices[0].players) == len(action_choices[0].positions) == 0:
                action_choice = action_choices[0]
                action = botbowl.Action(action_choice.action_type)
                self.historical_real_action_sequence.append(action)
                return action
        # (4) Choose Use_Reroll rather than Not_use_roll
        if botbowl.ActionType.USE_REROLL in actions_type and botbowl.ActionType.DONT_USE_REROLL in actions_type and len(actions_type) == 2:
            action_choice = game.state.available_actions[0] # USE_Reroll
            action = botbowl.Action(action_choice.action_type)
            self.historical_real_action_sequence.append(action)
            return action
        # (5) Priority is given to STAND_UP
        if botbowl.ActionType.STAND_UP in actions_type:
            for action_choice in game.state.available_actions:
                if action_choice.action_type == botbowl.ActionType.STAND_UP:
                    action = botbowl.Action(action_choice.action_type)
                    self.historical_real_action_sequence.append(action)
                    return action
        
        # # (6) Randomly place ball
        if botbowl.ActionType.PLACE_BALL in actions_type:
            action_choice = game.state.available_actions[0]
            position = np.random.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
            player = np.random.choice(action_choice.players) if len(action_choice.players) > 0 else None
            action = botbowl.Action(action_choice.action_type, position=position, player=player)
            self.historical_real_action_sequence.append(action)
            return action
        
        action = self.getBestAction(game) # take as input the game environment
        return action

    def end_game(self, game):
        pass

# Register the bot to the framework
botbowl.register_bot('mcts-bot', MCTSBot)

if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    # config = botbowl.load_config("bot-bowl")
    config = botbowl.load_config("web")
    config.competition_mode = False
    # config.pathfinding_enabled = False
    config.pathfinding_enabled = True
    config.debug_mode = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    # Play game
    wins = 0
    draws = 0
    TDs = 0
    games_num = 1
    for i in range(games_num):
        away_agent = botbowl.make_bot("random")
        home_agent = botbowl.make_bot("mcts-bot")

        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True
        
        print("Starting game", (i+1))
        game.init()
        print("*******************************************************************Game is over*******************************************************************")
        if game.get_winning_team() is game.state.home_team:
            wins += 1
        elif game.get_winning_team() is None:
            draws += 1
        TDs += game.state.home_team.state.score
        print(f"Wins: {wins}/{i+1}, Draws: {draws}/{i+1}, Loses: {i+1-wins-draws}/{i+1}")
        print(f"Own TDs per game={TDs/(i+1)}")
        print()
        print()
    print(f"Match over! Wins: {wins}/{games_num}, Draws: {draws}/{games_num}, Loses: {games_num-wins-draws}/{games_num}")
    print(f"Own TDs per game={TDs/games_num}")