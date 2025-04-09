class CellType:
    WALL = "#"  # character for wall
    ONE_TEMP = "*"  # character for player one temp barrier
    ONE_PERM = "X"  # character for player one barrier
    TWO_TEMP = "."  # character for player two temp barrier
    TWO_PERM = "O"  # character for player two barrier
    SPACE = " "  # character for space
    WHITE_WALKER = "W"  # character for white-walker
    DEATH = "~"  # position to show when a player is killed

    STOPS_WHITE_WALKERS = [WALL, ONE_PERM, TWO_PERM, WHITE_WALKER] # defines which cell types block WW movement
