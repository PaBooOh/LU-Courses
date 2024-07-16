import numpy as np
import botbowl
from botbowl.core import Action

C = 1.0 / np.sqrt(2) # Temperature coeff for UCB

# Node
class MCTSNode():

    """
    :param
    parent: the parent node of current node
    children: the children nodes of current node
    visit_num: number of visits to the current node
    Q: win rate of current node
    win_num: number of wins of current node
    action: the action represented by the current node
    team: the team performs this action (home_team or away_team)
    state: forward steplist
    action_player: the player performs this action
    """
    def __init__(self, action = None, state_steps = None, team = None, parent = None, action_player = None):
        self.parent = parent
        self.children = []
        self.visit_num = 0
        self.Q = 0.0
        self.win_num = 0
        self.action = action
        self.team = team
        self.state = state_steps
        self.action_player = action_player

    # get/set
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

    """
    Calculate the UCB:
    check if the team of the child correspond to the team of the root due to the fact that
    actions are not necessarily performed alternate in this game.
    """
    def getUCB(self, C):
        left_term = self.getQ() if self.team == self.parent.team else (1.0 - self.getQ())
        right_term = C * np.sqrt(2.0 * np.log(self.getParent().getVisitNum()) / self.getVisitNum())
        UCB_value = left_term + right_term
        return UCB_value


    """
    Selection phrase of MCTS
    """
    def select(self):
        selected_node = max(self.children, key=lambda child: child.getUCB(C))
        return selected_node
        
    """
    Expand phrase of MCTS
    """
    def expand(self, game, root_node, historical_real_action_sequence):
        leaf_node_state_id = game.get_step() # store current state (step) that represents the leaf node to do "expand"
        available_actions = game.get_available_actions()
        actions_type = [action_choice.action_type for action_choice in game.state.available_actions]
        ball_carrier = game.get_ball_carrier()
        # To reduce huge MOVE space.
        surround_positions = None
        if game.get_ball_position() is not None:
            surround_positions = [
                botbowl.Square(game.get_ball_position().x - 1, game.get_ball_position().y - 1),
                botbowl.Square(game.get_ball_position().x + 1, game.get_ball_position().y - 1),
                botbowl.Square(game.get_ball_position().x - 1, game.get_ball_position().y + 1),
                botbowl.Square(game.get_ball_position().x + 1, game.get_ball_position().y + 1),
                botbowl.Square(game.get_ball_position().x + 1, game.get_ball_position().y),
                botbowl.Square(game.get_ball_position().x - 1, game.get_ball_position().y),
                botbowl.Square(game.get_ball_position().x, game.get_ball_position().y + 1),
                botbowl.Square(game.get_ball_position().x, game.get_ball_position().y - 1)
                ]
        
        print('Ball_carrier: ', ball_carrier)
        print('>>Leaf node available actions: ', available_actions)
        print('===========================================')
        # Heuristic
        # (1) Formation setup is not allowed to use custom setup.
        if botbowl.ActionType.PLACE_PLAYER in actions_type: # 
            # if foramtion is selceted but have yet to end setup
            formations = [botbowl.ActionType.SETUP_FORMATION_LINE, botbowl.ActionType.SETUP_FORMATION_SPREAD, botbowl.ActionType.SETUP_FORMATION_WEDGE, botbowl.ActionType.SETUP_FORMATION_ZONE]
            if self.action.action_type in formations:
                action_choice = game.state.available_actions[1] # END_SETUP
                team = action_choice.team
                action = Action(action_choice.action_type)
                game.step(action)
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
        
        # (2) Priority is given to STAND_UP
        if botbowl.ActionType.STAND_UP in actions_type:
            for action_choice in game.state.available_actions:
                if action_choice.action_type == botbowl.ActionType.STAND_UP:
                    team = action_choice.team
                    action = Action(action_choice.action_type)
                    game.step(action)
                    next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                    node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                    self.children.append(node_expanded)
                    print('>>Expand an action: ', action_choice.action_type)
                    return
        # (3) Priority is given to find and pick up the ball if available actions like START_? and ball is on the ground.
        if game.get_ball_carrier() is None and botbowl.ActionType.START_MOVE in actions_type:
            print('Start to find ball...')
            for action_choice in available_actions:
                if botbowl.ActionType.START_MOVE == action_choice.action_type:
                    for player in action_choice.players:
                        team = action_choice.team
                        action = Action(action_choice.action_type, player=player)
                        game.step(action) # perform an action and then get to next state
                        next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                        node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self, action_player=player)
                        self.children.append(node_expanded)
                        print('>>Expand an action that finds the ball: ', action_choice.action_type)
                    return
        # if available actions contain MOVE and previous action is START_MOVE and the ball is on the ground
        if game.get_ball_carrier() is None and botbowl.ActionType.MOVE in actions_type:
            if (historical_real_action_sequence[-1].action_type == botbowl.ActionType.START_MOVE and len(root_node.children) == 0) or (self.action is not None and self.action.action_type == botbowl.ActionType.START_MOVE):
                print('Start to find ball...')
                ball_reachable = False
                for action_choice in available_actions:
                    if botbowl.ActionType.MOVE == action_choice.action_type:
                        for position in action_choice.positions:
                            if position == game.get_ball_position():
                                ball_reachable = True
                                team = action_choice.team
                                action = Action(action_choice.action_type, position=position)
                                game.step(action) # perform an action and then get to next state
                                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                                self.children.append(node_expanded)
                                print('>>Expand an action that picks up the ball: ', action_choice.action_type)
                        if ball_reachable:
                            return
        
        # ========================= Start to expand actions =======================
        for action_choice in available_actions:
            team = action_choice.team # need to get which team will perforam this action for backup phrase.
            # Heuristic
            if botbowl.ActionType.UNDO == action_choice.action_type: # (1) ignore UNDO
                continue
            if botbowl.ActionType.MOVE == action_choice.action_type:
                # (2) More than 3 (included) consecutive actions MOVE are not allowed.
                if self.action is not None and self.action.action_type == botbowl.ActionType.MOVE or (historical_real_action_sequence[-1].action_type == botbowl.ActionType.MOVE and len(root_node.children) == 0): 
                    print('Ignore 3 consecutive MOVE.')
                    continue
                # (3) Make a surrounding formation around the ball_carrier to reduce the huge size of tree.
                if (len(root_node.children) == 0 and historical_real_action_sequence[-1].action_type == botbowl.ActionType.START_MOVE) or (self.action is not None and botbowl.ActionType.START_MOVE == self.action.action_type):
                    if ball_carrier is not None and surround_positions is not None:
                        reach_surrounding_or_carrier = False
                        if ball_carrier != self.action_player: # Ball carrier has no need to form surrounding formation.
                            print('>>> Start find suitable surrounding position !!!')
                            for position in action_choice.positions:
                                if position in surround_positions:
                                    reach_surrounding_or_carrier = True
                                    print('>>> Get surrounding position !!!')
                                    action = Action(action_choice.action_type, position=position)
                                    game.step(action) # perform an action and then get to next state
                                    next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                                    node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                                    self.children.append(node_expanded)
                        else:
                            reach_surrounding_or_carrier = True
                            for position in action_choice.positions:
                                action = Action(action_choice.action_type, position=position)
                                game.step(action) # perform an action and then get to next state
                                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self)
                                self.children.append(node_expanded)
                            print('>>> Ball carrier moved !!!')
                        if reach_surrounding_or_carrier:
                                return
                                    
            print('>>Expand an action: ', action_choice.action_type)
            print('>>Action.Players_len: ', len(action_choice.players))
            print('>>Action.Positions_len: ', len(action_choice.positions))
            if botbowl.ActionType.DONT_USE_REROLL == action_choice.action_type and len(actions_type) == 2: # (4) ignore dont use reroll
                continue
            
            # E.g., Actions: Start_move, Start_block etc
            for player in action_choice.players:
                # Heuristic
                # (4) Players who are not holding ball is not allowed to perform Start_handoff/pass.
                if botbowl.ActionType.START_HANDOFF == action_choice.action_type or botbowl.ActionType.START_PASS == action_choice.action_type:
                    if player != game.get_ball_carrier():
                        continue
                # (5) # Players not in tackle zone is not allowed to start block.
                if botbowl.ActionType.START_BLOCK == action_choice.action_type:
                    if game.num_tackle_zones_in(player) == 0:
                        continue
                action = Action(action_choice.action_type, player=player)
                game.step(action) # perform an action and then get to next state
                next_state = game.revert(leaf_node_state_id) # revert to the previous state and get the next_state.
                node_expanded = MCTSNode(action=action, state_steps=next_state, team=team, parent=self, action_player=player)
                self.children.append(node_expanded)

            # Expand an action with its corresponding player/position
            # E.g., Actions: move, block etc
            for position in action_choice.positions:
                action = Action(action_choice.action_type, position=position)
                game.step(action) # perform an action and then get to next state
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
            
            print('===========================================')
            
        print('End of expansion.')
        print()

    """
    Simulation phrase
    """
    def simulate(self, game):
        # print('Start simulation...')
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
            
            action_choice = np.random.choice(game.state.available_actions)
            position = np.random.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
            player = np.random.choice(action_choice.players) if len(action_choice.players) > 0 else None
            action = botbowl.Action(action_choice.action_type, position=position, player=player)
            game.step(action)
        
        winner_team = game.get_winning_team()
        
        self.backpropagate(winner_team) # (4) Backup: takes as input the terminal game

    def backpropagate(self, winner_team):
        # print('Start backup...')
        current_team = self.team
        # Once the terminal is reached, update the result starting from the expanded node.
        # (1) update the expanded node or the terminal node
        self.visit_num += 1
        if winner_team is current_team: 
            self.win_num += 1
        self.Q = 1.0 * self.win_num / self.visit_num # Q represents Winning Rate here

        # (2) reversely recursive update on previous (parent) of the expanded node until the root node is reached
        node = self.parent # recursion
        while node:
            node.visit_num += 1
            if winner_team is node.team: 
                node.win_num += 1
            node.Q = 1.0 * node.win_num / node.visit_num # Q represents Winning Rate here
            node = node.parent