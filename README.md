# tictactoe_ai

This code is a Python implementation of a Tic-Tac-Toe game with an AI agent. It uses NumPy for array operations, random for generating random numbers, and pickle for saving and loading the AI's knowledge (brain). The code is divided into two main classes: `TicTacToe` and `Agent`.

### `TicTacToe` Class:

- **Initialization (`__init__`)**: Sets up the game state as a 9-element NumPy array initialized to zeros, representing the empty spaces on the board. It also initializes the winner to `None` and the current player to `1` (X).
- **`create_current_game` Method**: Prints the current game state in a user-friendly format, showing the board with X, O, and empty spaces.
- **`get_current_game` and `get_current_game_tuple` Methods**: Return the current game state as a NumPy array and as a tuple, respectively.
- **`get_available_positions` Method**: Returns the indices of the empty spaces on the board.
- **`reset_game` Method**: Resets the game state to a new game with all spaces empty and the current player set to `1`.
- **`get_player` Method**: Returns the current player.
- **`make_move` Method**: Makes a move for the current player at the specified action (index) if it's an available position, and switches the player.
- **`_make_move` Method**: Similar to `make_move` but intended for internal use.
- **`get_next_states` Method**: Returns all possible next states from the current state.
- **`is_winner` Method**: Checks if there's a winner or if the game is a draw, updates the winner, and resets the game if necessary.

### `Agent` Class:

- **Initialization (`__init__`)**: Sets up the agent with a game instance, player symbol, and training parameters like number of episodes, epsilon (for exploration vs. exploitation), discount factor (for future rewards), and epsilon reduction factor.
- **`save_brain` and `load_brain` Methods**: Save and load the agent's knowledge (Q-values for each state-action pair) using pickle.
- **`reward` Method**: Updates the Q-values based on the result of a game and the move history.
- **`use_brain` Method**: Decides the best action to take based on the current game state and the agent's knowledge.
- **`train_brain_x_byrandom` and `train_brain_o_byrandom` Methods**: Train the agent's knowledge by playing a specified number of episodes, where the agent makes decisions based on its current knowledge and explores new actions based on epsilon.
- **`play_with_user` Method**: Allows the user to play against the agent, using the agent's knowledge to decide moves.


### Q-Learning

The `Agent` class uses a dictionary (`self.brain`) to store the Q-values. This dictionary maps a tuple of the current game state and action to a Q-value. This approach is somewhat similar to a Q-table but uses a more flexible data structure.

How it is implemented:

- **Initialization**: When an `Agent` instance is created, its `brain` attribute is initialized as an empty dictionary. This is where the Q-values will be stored.
- **Saving and Loading Q-values**: The `save_brain` and `load_brain` methods use the `pickle` module to serialize and deserialize the `brain` dictionary to and from a file. This allows the agent to retain its learned knowledge between sessions.
- **Updating Q-values**: The `reward` method updates the Q-values based on the outcome of a game and the move history. It iterates through the move history, reversing it to start from the current state and working backwards. For each state-action pair, it updates the Q-value based on the reward received and the discount factor, which reflects the importance of future rewards.
- **Choosing Actions**: The `use_brain` method selects the best action to take based on the current game state. It iterates through all available actions, calculates the Q-value for each action, and selects the action with the highest Q-value. If there are multiple actions with the same highest Q-value, it randomly chooses one.

This code demonstrates reinforcement learning through Q-learning, where the agent learns to play Tic-Tac-Toe by playing games with itself (with random opponent) and updating its knowledge based on the outcomes.
