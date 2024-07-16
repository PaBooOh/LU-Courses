#!/usr/bin/env python3

import botbowl
import numpy as np
from copy import deepcopy
from botbowl.core import Action, ActionType
from Model import UCT

class MCTSAgent(botbowl.Agent):
    """
    :param
    my_team: Agent-controlled team
    rnd: random seed
    formation_setup: Is the formation set
    """
    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        self.formation_setup = False


    def new_game(self, game, team):
        self.my_team = team

    """
    Find the next best action using MCTS search
    """
    def SearchBestAction(self, game):
        gameCopy = deepcopy(game)
        # enable the forward model
        gameCopy.enable_forward_model()
        gameCopy.home_agent.human = True
        gameCopy.away_agent.human = True

        # mcts search
        mcts = UCT(gameCopy)
        action = mcts.UCTSearch()
        return action

    """
    Get the type of available actions in the current state
    """
    def getAvailableActionTypes(self, game):
        actionChoices = game.state.available_actions
        actionTypes = [actionChoice.action_type for actionChoice in actionChoices]
        return actionTypes

    """
    Specify fixed actions according to the game rules
    """
    def SpecifyFixedAction(self, game):
        actionChoices = game.state.available_actions
        actionTypes = self.getAvailableActionTypes(game)
        # heads/tails
        if botbowl.ActionType.HEADS in actionTypes or botbowl.ActionType.RECEIVE in actionTypes:
            actionChoice = self.rnd.choice(actionChoices)
            return Action(actionChoice.action_type)
        # select a formation
        if botbowl.ActionType.PLACE_PLAYER in actionTypes and self.formation_setup == False:
            actionChoice = actionChoices[-1]
            self.formation_setup = True
            return Action(actionChoice.action_type)
        # end formation select
        if botbowl.ActionType.PLACE_PLAYER in actionTypes and self.formation_setup == True:
            self.formation_setup = False
            return Action(ActionType.END_SETUP)
        # end turn...
        if len(actionTypes) == 1:
            if len(actionChoices[0].players) == len(actionChoices[0].positions) == 0:
                actionChoice = actionChoices[0]
                return Action(actionChoice.action_type)
        # use reroll
        if botbowl.ActionType.USE_REROLL in actionTypes and botbowl.ActionType.DONT_USE_REROLL in actionTypes and len(
                actionTypes) == 2:
            return Action(ActionType.USE_REROLL)

        return False

    """
    is called at every step in the game where the agent is supposed to perform an action.
    """
    def act(self, game):
        action = self.SpecifyFixedAction(game)
        if action == False:
            action = self.SearchBestAction(game)
        print("the best action is:", action)
        return action

    """
    end game
    """
    def end_game(self, game):
        pass


