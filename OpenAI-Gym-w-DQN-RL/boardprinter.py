#!/usr/bin/python

from GoT_types import CellType

class TextColors:
    """
    Defines stylistic constants
    """
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


color_dict = {
    CellType.ONE_TEMP: TextColors.RED,
    CellType.ONE_PERM: TextColors.RED,
    CellType.TWO_TEMP: TextColors.BLUE,
    CellType.TWO_PERM: TextColors.BLUE,
    CellType.WHITE_WALKER: TextColors.YELLOW,
    CellType.WALL: TextColors.BOLD,
}

class BoardPrinter:
    @staticmethod
    def state_to_string(state, colored):
        """
        Input:
            state- GoTState to stringify
            colored- boolean. if true, use color
        Output:
            Returns a string representing a readable version of the state.
        """

        if colored:
            return "{}".format(
                BoardPrinter._board_to_pretty_string_colored(state),
            )
        else:
            return BoardPrinter._board_to_pretty_string(state.board)

    @staticmethod
    def _board_to_pretty_string(board):
        """
        Converts a board from its CellType representation to a descriptive string
        """
        s = ""
        for row in board:
            for cell in row:
                s += cell
            s += "\n"
        return s

    @staticmethod
    def _board_to_pretty_string_colored(state):
        """
        Converts a board from its class representation to a descriptive string which includes color descriptors
        """
        s = ""
        for row in state.board:
            for cell in row:
                s += BoardPrinter._colored_character(cell)
            s += "\n"
        return s

    @staticmethod
    def _colored_character(cell):
        """
        Input:
            cell- CellType
        Output:
            A three part cell representation including color desciptors, if avaliable
        """
        color = None
        if cell in color_dict:
            color = color_dict[cell]
        else:
            return cell
        return "{}{}{}".format(color, cell, TextColors.ENDC) if color else cell
