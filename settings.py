"""Constants etc..."""
from random import choice


class Rewards:
    """
    Rewards for actions taken by an agent:

        invalid - try to move to blocked cell or out of maze
        visited - move to already visited cell
        valid   - move to available cell
        success - exit found
    """
    INVALID, VISITED, VALID, SUCCESS = -1.0, -0.25, 0., 1.

    MIN_REWARDS = -10  # agent loses if total rewards < MIN_REWARDS


class Moves:
    """
    Possbile moves:

        LEFT  = 0
        RIGHT = 1
        UP    = 2
        DOWN  = 3

    Position is given by numpy 2D array indices [row, col] -> [y, x]
    """
    ALL = LEFT, RIGHT, UP, DOWN = range(4)

    @staticmethod
    def move(position, direction):
        """Return new position"""
        if direction == Moves.LEFT:    return position[0], position[1] - 1                         
        elif direction == Moves.RIGHT: return position[0], position[1] + 1
        elif direction == Moves.UP:    return position[0] - 1, position[1]
        elif direction == Moves.DOWN:  return position[0] + 1, position[1]

    @staticmethod
    def random_move(position):
        """Make a move in random direction"""
        return Moves.move(position, choice(Moves.ALL))
