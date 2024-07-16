#!/usr/bin/env python3

import botbowl
from Agent import MCTSAgent
import time

"""
Register, initialise and play the game
"""
def main():
    # Register the agent to the botbowl framework
    botbowl.register_bot('mcts-agent', MCTSAgent)
    print("the mcts agent has been registered...")
    # Setup the game
    config = botbowl.load_config("web")
    # config.pathfinding_enabled = False
    # print("pathfinding_enabled: ", config.pathfinding_enabled)
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    # play games
    # Statistic game results and game time
    win = 0
    draw = 0
    loss = 0

    game_times = 1

    games_start = time.time()
    for t in range(game_times):
        game_start = time.time()
        home_agent = botbowl.make_bot("mcts-agent")
        away_agent = botbowl.make_bot("random")
        game = botbowl.Game(t, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)

        print("Game {} start....".format(t+1))
        game.init()
        print("Game {} over.....".format(t+1))

        if game.get_winning_team() is game.state.home_team:
            win += 1
            print("Game {} win!".format(t+1))

        if game.get_winning_team() is None:
            draw += 1
            print("Game {} draw".format(t+1))


        if game.get_winning_team() is game.state.away_team:
            loss += 1
            print("Game {} loss".format(t+1))

        game_end = time.time()
        print("game time: ", game_end - game_start)
        print("The number of touchdown is: ", game.state.home_team.state.score)
        print("-" * 60)

    games_end = time.time()

    print("It takes {} seconds to play {} games.".format(games_end-games_start, game_times))
    print(f"won {win}/{game_times}")
    print(f"draw {draw}/{game_times}")
    print(f"loss {loss}/{game_times}")


if __name__ == "__main__":
    main()


