import time
from GoT_problem import GoTProblem
import copy, argparse, signal
from collections import defaultdict
import support
import traceback

def run_game(asp, bots, visualizer=None, delay=.2, max_wait=1, max_rounds=500, colored=True):
    """
    Inputs:
        - asp: an adversarial search problem
        - bots: a list in which the i'th element is the bot for player i
        - visualizer (optional): a void function that takes in a game state
          and does something to visualize it. If no visualizer argument is
          passed, run_game will not visualize games.
    Runs a game and outputs the evaluation of the terminal state.
    This function uses the signal module, so it can only be called on unix-based machines
    """
    state = asp.get_start_state()
    tmp_rounds = 0
    if not visualizer == None:
        visualizer(state, colored)

    while not (asp.is_terminal_state(state)):
        exposed = copy.deepcopy(asp)
        signal.signal(signal.SIGALRM, support.timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, max_wait)
        try:
            # run AI
            decision = bots[state.ptm].decide(exposed)
        except support.TimeoutException:
            print(traceback.format_exc())
            if visualizer:
                print(
                    """Warning. Player %s took too long to decide on a move.
                    They will go UP this round."""
                    % (state.ptm + 1)
                )
            decision = "U"
        except: # pylint: disable=bare-except
            if visualizer:
                print(
                    """Warning. The move for player %s encountered an unexpected error.
                    They will go UP this round."""
                    % (state.ptm + 1)
                ) 
            decision = "U"
        
        signal.setitimer(signal.ITIMER_REAL, 0)

        available_actions = asp.get_available_actions(state)
        if not decision in available_actions:
            decision = list(available_actions)[0]

        result_state = asp.transition_runner(state, decision)
        asp.set_start_state(result_state)

        state = result_state
        if not visualizer == None:
            visualizer(state, colored)
            time.sleep(delay)
        
        tmp_rounds += 1
        if tmp_rounds == (2*max_rounds) and not (asp.is_terminal_state(state)):
            asp.intercept_max_rounds(state)

    return asp.evaluate_state(asp.get_start_state())


def run_game_training(asp, bots, visualizer=None, delay=0.2, max_wait=None, max_rounds=500, colored=True):
    """
    Inputs:
        - asp: an adversarial search problem
        - bots: a list in which the i'th element is the bot for player i
        - visualizer (optional): a void function that takes in a game state
          and does something to visualize it. If no visualizer argument is
          passed, run_game will not visualize games.
    Runs a game and outputs the evaluation of the terminal state. 
    This function does not utilize time controls and can be run on windows machines.
    """
    state = asp.get_start_state()
    tmp_rounds = 0
    if not visualizer == None:
        visualizer(state, colored)

    while not (asp.is_terminal_state(state)):
        exposed = copy.deepcopy(asp)
        try:
            # run AI
            decision = bots[state.ptm].decide(exposed)
        except: # pylint: disable=bare-excepte
            print(traceback.format_exc())
            if visualizer:
                print(
                    """Warning. The move for player %s encountered an unexpected error.
They will go UP this round."""
                    % (state.ptm + 1)
                ) 
            decision = "U"

        available_actions = asp.get_available_actions(state)
        if not decision in available_actions:
            decision = list(available_actions)[0]

        result_state = asp.transition_runner(state, decision)
        asp.set_start_state(result_state)

        state = result_state
        if not visualizer == None:
            visualizer(state, colored)
            time.sleep(delay)
        
        tmp_rounds += 1
        if tmp_rounds == (2*max_rounds) and not (asp.is_terminal_state(state)):
            asp.intercept_max_rounds(state)

    return asp.evaluate_state(asp.get_start_state())

def run_decision(bots, curr_state, curr_exposed, return_state):
        decision = bots[curr_state.ptm].decide(curr_exposed)
        return_state["decision"] = decision

def main():
    # Feel free to uncomment these if you want to test deterministic behavior!
    # random.seed(1)
    # np.random.seed(1)

    parser = argparse.ArgumentParser(prog="gamerunner", usage="%(prog)s [options]")
    parser.add_argument(
        "-map", type=str, help=HelpMessage.MAP, default=support.Argument_Defaults.MAP
    )
    parser.add_argument(
        "-max_wait",
        type=float,
        help=HelpMessage.MAX_WAIT,
        default=support.Argument_Defaults.MAX_WAIT,
    )
    parser.add_argument(
        "-max_rounds",
        type=str,
        nargs="+",
        help=HelpMessage.MAX_ROUNDS,
        default=support.Argument_Defaults.MAX_ROUNDS,
    )
    parser.add_argument(
        "-bots",
        type=str,
        nargs="+",
        help=HelpMessage.BOTS,
        default=support.Argument_Defaults.BOTS,
    )
    parser.add_argument(
        "-runner",
        type=str,
        help=HelpMessage.RUNNER,
        default=support.Argument_Defaults.RUNNER
    )
    parser.add_argument(
        "-image_delay",
        type=float,
        help=HelpMessage.IMAGE_DELAY,
        default=support.Argument_Defaults.IMAGE_DELAY,
    )
    parser.add_argument(
        "-no_image", dest="show_image", help=HelpMessage.NO_IMAGE, action="store_false"
    )
    parser.set_defaults(show_image=True)
    parser.add_argument("-multi_test", type=int, help=HelpMessage.MULTI_TEST)
    parser.add_argument(
        "-no_color", dest="colored", help=HelpMessage.COLOR, action="store_false"
    )
    parser.add_argument(
        "-no_msg", dest="print_message", help=HelpMessage.NO_MSG, action="store_false"
    )
    parser.set_defaults(colored=True)

    args = parser.parse_args()

    wait = args.max_wait
    rounds = args.max_rounds
    bots = support.determine_bot_functions(args.bots)
    delay = args.image_delay
    verbose = args.show_image
    multi = args.multi_test
    colored = args.colored
    runner = args.runner
    print_msg = args.print_message

    visualizer = None
    if verbose:
        visualizer = GoTProblem.visualize_state  
  
    if runner == "training":
        runner_f = run_game_training
    else:
        runner_f = run_game

    if multi is not None:
        winners = defaultdict(int)
        bots = support.determine_bot_functions(args.bots)
        for i in range(multi):
            game = GoTProblem(args.map, i % 2, message_print=print_msg)
            outcome = runner_f(game, bots, visualizer, delay, wait, rounds, colored)
            winner = outcome.index(1)
            winners[winner] += 1
            for bot in bots:
                bot.cleanup()
        for winner, wins in list(winners.items()):
            print("Player %s won %d out of %d times" % (winner + 1, wins, multi))

    else:
        game = GoTProblem(args.map, 0, message_print=print_msg)
        outcome = runner_f(game, bots, visualizer, delay, wait, rounds, colored)
        winner = outcome.index(1) + 1
        print("Player %s won!" % winner)


class HelpMessage:
    MAP = (
        '''the filename of the map to use for this game.
        Defaults to "'''
        + support.Argument_Defaults.MAP
        + """."""
    )
    MAX_WAIT = (
        """The amount of time (in seconds) the game engine will wait
        for a player to decide what move they want to make. If the player takes too long,
        they go north. Defaults to """
        + str(support.Argument_Defaults.MAX_WAIT)
        + """ (this
        will be reset during grading). If you want to play around with keyboard inputs, you may
        set it higher in support.py."""
    )
    BOTS = (
        '''which bot each player will use. Valid bot types include "student", "manual", 
        "attack", "random", "ta1", "ta2". This argument takes in a sequence of bot types,
        where the first bot is used for the first player, the second bot is for the second
        player, and so on. Defaults to "'''
        + support.Argument_Defaults.BOTS[0]
        + """
         """
        + support.Argument_Defaults.BOTS[1]
        + """". Note that errors will occur if there
        are not enough AIs for the number of players on the board."""
    )
    IMAGE_DELAY = (
        """The amount of time (in seconds) to wait after printing the current
        state of the game. This is just to give users more time to watch the game progress.
        Defaults to """
        + str(support.Argument_Defaults.IMAGE_DELAY)
        + """."""
    )
    NO_IMAGE = """include this flag (with no arguments) to suppress the output of all
        board states and just get final results"""
    MULTI_TEST = """Test this map (multi_test) times in a row. Useful if you want to see how
        randomized algorithms do on average. It's recommended that you turn off verbose for this.
        Tracks how many times each player won across all games."""
    MAX_ROUNDS = ("""The maximum number of rounds for two players in a game. One round corresponds a move 
        of player 1 and a move of player 2. If the maximum rounds arg is reached but no winning condition
        satisfied, the winner is the player with more space. If two player claim the same size of space,
        a random player wins. defaults to"""
        + str(support.Argument_Defaults.MAX_ROUNDS)
        + """."""
    )
    COLOR = """include this flag to remove the coloration in the output of board states. Use this if
        the coloring does not display properly"""
    RUNNER = ("""include this flag to change the timeout behavior for the game runner. use unix for 
        mac/unix based machines; training to set no time out behavior 
        (and that's the only way to run on windows); defaults to """
        + str(support.Argument_Defaults.RUNNER)
        + """."""
    )
    NO_MSG = ("""include this flag if you don't want any terminal outputs, especially running multi 
    games or in the training process""")


if __name__ == "__main__":
    main()
