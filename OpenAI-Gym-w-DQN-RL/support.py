########################### support.py #############################
"""
STUDENT INFO: You may want to add new elif statements to support new bots

This file just contains a few support functions used by the other
files

"""
import bots
import tabots


class Argument_Defaults:
    MAP = "./maps/large_room.txt"
    MAX_WAIT = 1.0
    BOTS = ["random", "random"]
    IMAGE_DELAY = 0.2
    RUNNER = "unix"
    MAX_ROUNDS = 500


def determine_bot_functions(bot_names):
    bot_list = []
    for name in bot_names:
        if name == "student":
            bot_list.append(bots.StudentBot())
        elif name == "manual":
            bot_list.append(bots.ManualBot())
        elif name == "random":
            bot_list.append(bots.RandBot())
        elif name == "attack":
            bot_list.append(bots.AttackBot())
        elif name == "safe":
            bot_list.append(bots.SafeBot())
        elif name == "ta1":
            bot_list.append(tabots.TA1Bot())
        elif name == "ta2":
            bot_list.append(tabots.TA2Bot())
        else:
            raise ValueError(
                """Bot name %s is not supported. Value names include "student", "manual", 
                "random", "safe", "attack", "ta1", "ta2" """
                % name
            )
    return bot_list


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Player action timed out.")