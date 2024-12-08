from simulator.game.connect import Action  # type: ignore[import]

# ActionPolicy represents a probability distribution over available actions in a given state.
# It maps each possible action to its probability of being selected, providing a strategy
# for action selection based on the current game state.
ActionPolicy = dict[Action, float]


# Value represents the estimated value of a game state for each player.
# It is a list of floating-point numbers, where each element corresponds
# to the expected outcome or utility for a specific player in the current game state.
# The list's length matches the number of players in the game.
Value = list[float]
