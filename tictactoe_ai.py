import numpy as np
import random
import pickle

class TicTacToe:
    
    def __init__(self):
        self.current_state = np.zeros(9, dtype = np.int8)
        self.winner = None
        self.player = 1
    
    def create_current_game(self):
        current_state = ['X' if x == 1 else 'O' if x == -1 else '--' for x in self.current_state]
        print("LEGEND:")
        print(' 1     2     3 ')
        print(' 4     5     6 ')
        print(' 7     8     9 ')
        print("###############")
        print('\n')
        print(f'{current_state[0]:^5} {current_state[1]:^5} {current_state[2]:^5}')
        print(f'{current_state[3]:^5} {current_state[4]:^5} {current_state[5]:^5}')
        print(f'{current_state[6]:^5} {current_state[7]:^5} {current_state[8]:^5}')
        print('_'*15)

    
    def get_current_game(self):
        return self.current_state
    
    def get_current_game_tuple(self):
        return tuple(self.current_state)

    def get_available_positions(self):
        return (np.argwhere(self.current_state==0).ravel()) 

    def reset_game(self):
        self.current_state = np.zeros(9, dtype = np.int8)
        self.player = 1

    def get_player(self):
        return self.player

    def make_move(self, action): # player is 1 for X, player is -1 for O
        if action in self.get_available_positions():
            self.current_state[action] = self.player
            #self.create_current_game()
            self.player *= -1
        else:
            print('It is not available')

    def _make_move(self, _current_state, action): # it is not proper function for get input from user. should prefer make_move function
        _current_state[action] = self.player
        return _current_state

    def get_next_states(self):
        states = []
        _current_state = self.current_state
        _available_moves = self.get_available_positions()
        for move in _available_moves:
            states.append(self._make_move(_current_state = _current_state, action=move))
        return states 
    
    def is_winner(self, isgame = False):
        winner_coordinates = np.array([[0,1,2], [3, 4, 5], [6, 7, 8],
                                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                                [0, 4, 8], [2, 4, 6]])
        for coordinate in winner_coordinates:
            total = sum(self.current_state[coordinate])
            if total == 3: # X winner
                if isgame:
                    print('X is Winner!')
                self.winner = 1
                self.reset_game()
                return 1
            elif total == -3: # O Winner
                if isgame:
                    print('O is Winner!')
                self.winner = -1
                self.reset_game()
                return -1
            elif sum(self.current_state == 1) == 5: # Draw
                if isgame:
                    print('DRAW') 
                self.winner = -2
                self.reset_game()
                return -2
        return False

class Agent:

    def __init__(self, game, player = 'X', episode = 100000, epsilon = 0.9, discount_factor = 0.6, eps_reduce_factor = 0.01):

        self.game = game
        self.player = player
        self.brain = dict()
        self.episode = episode
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.results = {'X' : 0, 'O': 0, 'D': 0}
        self.eps_reduce_factor = eps_reduce_factor

    def save_brain(self, player):
        with open('brain'+player, 'wb') as brain_file:
            pickle.dump(self.brain, brain_file)

    def reward(self, player, move_history, result):
        _reward = 0
        
        if player == 1:
            if result == 1:
                _reward = 1
                self.results['X'] += 1 
            elif result == -1:
                _reward = -1
                self.results['O'] += 1 
                
        elif player == -1:
            if result == 1:
                _reward = -1
                self.results['X'] += 1 
            elif result == -1:
                _reward = 1
                self.results['O'] += 1
                
        if result == -2:
             self.results['D'] += 1
        move_history.reverse()
        
        for state, action in move_history:
            self.brain[state, action] = self.brain.get((state, action), 0.0) + _reward
            _reward *= self.discount_factor

    
