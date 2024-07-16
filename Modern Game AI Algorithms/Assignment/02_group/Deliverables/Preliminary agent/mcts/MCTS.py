from copy import deepcopy
import numpy as np
import botbowl
from botbowl.core import Action

C = 1.0 / np.sqrt(2) # Temperature coeff

# Node
class MCTSNode():
    def __init__(self, action = None, state_steps = None, team = None, parent = None):
        self.parent = parent
        self.children = []
        self.visit_num = 0
        self.Q = 0.0
        self.win_num = 0
        self.action = action
        self.team = team
        self.state = state_steps
    
    def getParent(self):
        return self.parent
    
    def getChildren(self):
        return self.children

    def getVisitNum(self):
        return self.visit_num
    
    def getQ(self):
        return self.Q
    
    def getAction(self):
        return self.action
    
    def isLeaf(self):
        return len(self.children) == 0
    
    def isRoot(self):
        return self.parent is None
    
    def isMid(self):
        return self.children != [] and self.parent is not None
    
    def getUCB(self, C):
        """
        Check if the team of the child correspond to the team of the root due to the fact that
        actions are not necessarily performed alternate in this game.
        """
        
        left_term = self.getQ() if self.team == self.parent.team else (1.0 - self.getQ())
        right_term = C * np.sqrt(2.0 * np.log(self.getParent().getVisitNum()) / self.getVisitNum())
        UCB_value = left_term + right_term
        return UCB_value

    # First phrase for MCTS playout
    def select(self):
        selected_node = max(self.children, key=lambda child: child.getUCB(C))
        return selected_node
        
    # Second phrase for MCTS playout
    def expand(self, game):
        leaf_node_state_id = game.get_step() # store current state (step) that represents the leaf node to do "expand"
        available_actions = game.get_available_actions()
        actions_type = [action_choice.action_type for action_choice in game.state.available_actions]

        print('>>Leaf node available actions: ', available_actions)
        # Custom rules
        # if botbowl.ActionType.PLACE_PLAYER == action_choice.action_type:
        #     formations = [botbowl.ActionType.SETUP_FORMATION_LINE, botbowl.ActionType.SETUP_FORMATION_SPREAD, botbowl.ActionType.SETUP_FORMATION_WEDGE, botbowl.ActionType.SETUP_FORMATION_ZONE]
        #     if self.action.action_type in formations:
        #        action_choice = game.state.available_actions[1] # End setup
        #        action = Action(action_choice.action_type)
        #        next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
        #     else:
        #         if self.action.action_type in formations:
        #             action = Action(action_choice.action_type, player=player)
        if botbowl.ActionType.PLACE_PLAYER in actions_type: # (1) Formation is not allowed to use custom
            # if foramtion is selceted but have yet to end setup
            formations = [botbowl.ActionType.SETUP_FORMATION_LINE, botbowl.ActionType.SETUP_FORMATION_SPREAD, botbowl.ActionType.SETUP_FORMATION_WEDGE, botbowl.ActionType.SETUP_FORMATION_ZONE]
            if self.action.action_type in formations:
                action_choice = game.state.available_actions[1] # END_SETUP
                team = action_choice.team
                action = Action(action_choice.action_type)
                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                self.children.append(node_expanded)
                print('>>Expand an action: ', action_choice.action_type)
            # if need to select an formation (Place players and end setup is not allowed in this case)
            elif self.action.action_type not in formations: 
                action_choice = np.random.choice(game.state.available_actions[2:])
                team = action_choice.team
                action = Action(action_choice.action_type)
                game.step(action)
                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                self.children.append(node_expanded)
                print('>>Expand an action: ', action_choice.action_type)
            return
        
        for action_choice in available_actions:
            # Custom rules
            if botbowl.ActionType.UNDO == action_choice.action_type: # (1) ignore UNDO
                continue
            if botbowl.ActionType.END_PLAYER_TURN == action_choice.action_type and len(actions_type) >= 2: # (2) ignore END_TURN if there are still other actions that can be chosen.
                continue
            if botbowl.ActionType.DONT_USE_REROLL == action_choice.action_type and len(actions_type) == 2:
                continue
            
            team = action_choice.team # need to get which team will perforam this action for backup phrase.
            print('>>Expand an action: ', action_choice.action_type)
            print('>>Action.Players_len: ', len(action_choice.players))
            print('>>Action.Positions_len: ', len(action_choice.positions))
            print('===========================================')
            print()
            
            # E.g., Actions: Start_move, Start_block etc
            for player in action_choice.players:
                action = Action(action_choice.action_type, player=player)
                # print(action_choice.action_type)
                game.step(action) # perform an action and then get to next state
                # next_state = game.get_step() # need to store state due to randomness of this game
                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                self.children.append(node_expanded)

            # E.g., Actions: move, block etc
            for position in action_choice.positions:
                action = Action(action_choice.action_type, position=position)
                # print(action_choice.action_type)
                game.step(action) # perform an action and then get to next state
                # next_state = game.get_step() # need to store state due to randomness of this game
                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                self.children.append(node_expanded)
            
            # E.g., Actions: End_turn, End_player_turn etc
            if len(action_choice.players) == len(action_choice.positions) == 0:
                action = Action(action_choice.action_type)
                game.step(action) # perform an action and then get to next state
                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                self.children.append(node_expanded)

    def simulate(self, game):
        game.forward(self.state) # from previous leaf node to this expanded node which is a new leaf node right now.
        # game = deepcopy(game)
        # Do a random simulation: rollout until the terminal is reached

        # randomly select an action as well as its corresponding player or position
        formation_setup = False
        while not game.state.game_over:
            actions_type = [action_choice.action_type for action_choice in game.state.available_actions]
            if botbowl.ActionType.USE_REROLL in actions_type and botbowl.ActionType.DONT_USE_REROLL in actions_type and len(actions_type) == 2:
                action_choice = game.state.available_actions[0]
                action = botbowl.Action(action_choice.action_type)
                game.step(action)
                continue
            if botbowl.ActionType.PLACE_PLAYER in actions_type and formation_setup == False:
                action_choice = np.random.choice(game.state.available_actions[2:]) # randomly choose a foramtion from the last two/one index
                formation_setup = True # formations is selected but have yet to end setup
                action = botbowl.Action(action_choice.action_type)
                game.step(action)
                continue
            if botbowl.ActionType.PLACE_PLAYER in actions_type and formation_setup == True:
                action_choice = game.state.available_actions[1] # END_SETUP
                formation_setup = False
                action = botbowl.Action(action_choice.action_type)
                game.step(action)
                continue
            if len(actions_type) == 1:
                action_choice = game.state.available_actions[0]
                if len(action_choice.players) == len(action_choice.positions) == 0:
                    action_choice = game.state.available_actions[0]
                    action =  botbowl.Action(action_choice.action_type)
                    game.step(action)
                    continue

            # while True:
            #     action_choice = np.random.choice(game.state.available_actions)
            #     if action_choice.action_type != botbowl.ActionType.PLACE_PLAYER:
            #         break
            
            action_choice = np.random.choice(game.state.available_actions)
            
            position = np.random.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
            player = np.random.choice(action_choice.players) if len(action_choice.players) > 0 else None
            action = botbowl.Action(action_choice.action_type, position=position, player=player)
            game.step(action)
        
        winner_team = game.get_winning_team()
        
        self.backpropagate(winner_team) # (4) Backup: takes as input the terminal game

    def backpropagate(self, winner_team):
        current_team = self.team
        # Once the terminal is reached, update the result starting from the expanded node.
        # (1) update the expanded node or the terminal node
        self.visit_num += 1
        if winner_team is current_team: 
            self.win_num += 1
        self.Q = 1.0 * self.win_num / self.visit_num # Q represents Winning Rate here

        # elif game.get_winning_team() is not current_team:
        #     pass
        # elif game.get_winning_team() is None:
        #     pass

        # (2) reversely recursive update on previous (parent) of the expanded node until the root node is reached
        node = self.parent # recursion
        while node:
            node.visit_num += 1
            if winner_team is node.team: 
                node.win_num += 1
            node.Q = 1.0 * node.win_num / node.visit_num # Q represents Winning Rate here
            node = node.parent