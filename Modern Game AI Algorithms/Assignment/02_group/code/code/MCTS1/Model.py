#!/usr/bin/env python3


import random
from botbowl.core import Action, ActionType
import math
from Node import MCTSNode
import botbowl
import numpy as np


class UCT():

    def __init__(self, game):
        self.computationBudget = 100 # the number of playouts
        self.game = game
        self.homeTeam = self.game.state.home_team
        self.awayTeam = self.game.state.away_team

    """
    Search for the best action
    """
    def UCTSearch(self):
        rootNode = MCTSNode()
        rootActions, teams = self.getAllAvailableActions()
        rootNode.setSubActions(rootActions)
        rootNode.setTeam(self.homeTeam)
        rootNode.setSubTeam(teams[0])
        root_id = self.game.get_step() # the current state id

        for i in range(self.computationBudget):
            # 1. Find the node to be extended
            expandNode = self.treePolicy(rootNode)
            # 2. Random action until the end of the game to get whether we win or not
            winTeam = self.defaultPolicy()
            # 3. Backpropagation
            self.backUp(expandNode, winTeam)
            # Revert the state
            self.game.revert(root_id)

        bestNextNode = self.bestChild(rootNode, False)
        action = bestNextNode.getAction()
        return action

    """
    Get all available actions in the current state and their corresponding team
    """
    def getAllAvailableActions(self):
        availableActions = []
        teams = []
        actionChoices = self.game.get_available_actions()
        for action in actionChoices:
            if len(action.players) != 0:
                for player in action.players:
                    a = Action(action.action_type, player=player)
                    availableActions.append(a)
                    teams.append(action.team)
            elif len(action.positions) != 0:
                for pos in action.positions:
                    a = Action(action.action_type, position=pos)
                    availableActions.append(a)
                    teams.append(action.team)
            else:
                a = Action(action.action_type)
                availableActions.append(a)
                teams.append(action.team)
        return availableActions, teams

    """
    Are all actions traversed once
    """
    def isAllExpanded(self, node): 
        if len(node.getSubActions()) == 0:
            return True
        else:
            return False


    """
    Selection phrase
    """
    def treePolicy(self, node):
        while not self.game.state.game_over:

            if self.isAllExpanded(node):
                node = self.bestChild(node, True)
                self.game.forward(node.getStepListFromLastS())
            else:
                #-------------- Preventing the game from falling into an setup formation, end_setup dead-end loop----------------#
                #-------------- And ignore the place_player action------------------------------------------------------------------#
                actions_type = [action_choice.action_type for action_choice in self.game.state.available_actions]
                if ActionType.PLACE_PLAYER in actions_type and node.isSetup == False:
                    newActionChoice = self.game.state.available_actions[-1]
                    newAction = Action(newActionChoice.action_type)
                    subNode = MCTSNode()
                    subNode.setParent(node)
                    subNode.setAction(newAction)
                    subNode.isSetup = True
                    subNode.setTeam(node.getSubTeam())

                    currentStateId = self.game.get_step()
                    self.game.step(newAction)
                    stepList = self.game.revert(currentStateId)
                    self.game.forward(stepList)
                    subNode.setStepListFromLastS(stepList)

                    subActions, teams = self.getAllAvailableActions()
                    subNode.setSubActions(subActions)
                    subNode.setSubTeam(teams[0])

                    node.subActions = []
                    node.setChildren(subNode)
                    return subNode

                if ActionType.PLACE_PLAYER in actions_type and node.isSetup == True:
                    newAction = Action(ActionType.END_SETUP)
                    subNode = MCTSNode()
                    subNode.setParent(node)
                    subNode.setAction(newAction)
                    subNode.isSetup = False
                    subNode.setTeam(node.getSubTeam())

                    currentStateId = self.game.get_step()
                    self.game.step(newAction)
                    stepList = self.game.revert(currentStateId)
                    self.game.forward(stepList)
                    subNode.setStepListFromLastS(stepList)

                    subActions, teams = self.getAllAvailableActions()
                    subNode.setSubActions(subActions)
                    subNode.setSubTeam(teams[0])

                    node.subActions = []
                    node.setChildren(subNode)
                    return subNode
                #-------------------------------------------------------------------------------------------------------
                subNode = self.expand(node)
                return subNode
        return node


    """
    Expandsion phrase
    """
    def expand(self, node): 
        availableActions = node.getSubActions()
        newAction = random.choice(availableActions)
        availableActions.remove(newAction)

        currentStateId = self.game.get_step()
        self.game.step(newAction)
        stepList = self.game.revert(currentStateId)
        self.game.forward(stepList)

        subNode = MCTSNode()
        subNode.setParent(node)
        subNode.setAction(newAction)
        subNode.setTeam(node.getSubTeam())
        subNode.setStepListFromLastS(stepList)

        if not self.game.state.game_over:
            subActions, teams = self.getAllAvailableActions()
            subNode.setSubActions(subActions)
            subNode.setSubTeam(teams[0])

        node.setChildren(subNode)
        return subNode

    """
    Simulation phrase
    """
    def defaultPolicy(self):
        formationSetup = False
        while not self.game.state.game_over: # True/False
            actionsType = [actionChoice.action_type for actionChoice in self.game.state.available_actions]
            if botbowl.ActionType.USE_REROLL in actionsType and botbowl.ActionType.DONT_USE_REROLL in actionsType and len(
                    actionsType) == 2:
                actionChoice = self.game.state.available_actions[0]
                action = botbowl.Action(actionChoice.action_type)
                self.game.step(action)
                continue
            if botbowl.ActionType.PLACE_PLAYER in actionsType and formationSetup == False:
                actionChoice = np.random.choice(self.game.state.available_actions[2:])
                formationSetup = True
                action = botbowl.Action(actionChoice.action_type)
                self.game.step(action)
                continue
            if botbowl.ActionType.PLACE_PLAYER in actionsType and formationSetup == True:
                actionChoice = self.game.state.available_actions[1]  # END_SETUP
                formationSetup = False
                action = botbowl.Action(actionChoice.action_type)
                self.game.step(action)
                continue
            if len(actionsType) == 1:
                actionChoice = self.game.state.available_actions[0]
                if len(actionChoice.players) == len(actionChoice.positions) == 0:
                    actionChoice = self.game.state.available_actions[0]
                    action = botbowl.Action(actionChoice.action_type)
                    # print(action)
                    self.game.step(action)
                    continue


            actionChoice = np.random.choice(self.game.state.available_actions)

            position = np.random.choice(actionChoice.positions) if len(actionChoice.positions) > 0 else None
            player = np.random.choice(actionChoice.players) if len(actionChoice.players) > 0 else None
            action = botbowl.Action(actionChoice.action_type, position=position, player=player)
            # print(action)
            self.game.step(action)

        winTeam = self.game.get_winning_team()
        return winTeam


    """
    backpropagation phrase
    """
    def backUp(self, subNode, winTeam):
        subNode.visitTimes += 1
        if winTeam is self.homeTeam:
            subNode.qualityValue += 1

        node = subNode.parent

        while node:
            node.visitTimes += 1
            if winTeam is self.homeTeam:
                node.qualityValue += 1
            node = node.parent

    """
    Calculate UCB and select best action
    """
    def bestChild(self, node, explorateOrNot):
        bestScore = 0
        bestSubNode = random.choice(node.getChildren())
        for subNode in node.getChildren():
            if explorateOrNot:
                C = 1 / math.sqrt(2.0)
            else:
                C = 0.0
            if subNode.getTeam() is self.homeTeam:
                leftTerm = subNode.getQualityValue() / subNode.getVisitTimes()
            else:
                leftTerm = 1 - (subNode.getQualityValue() / subNode.getVisitTimes())

            rightTerm = 2.0 * math.log(node.getVisitTimes()) / subNode.getVisitTimes()
            score = leftTerm + C * math.sqrt(rightTerm)

            if score > bestScore:
                bestSubNode = subNode
                bestScore = score
        return bestSubNode


"""
Tool method
"""
def printAvailableActionTypes(game):
    for actionChoice in game.get_available_actions():
        print(actionChoice.action_type.name, end=', ')
    print("\n", "-"*5, sep="")






























