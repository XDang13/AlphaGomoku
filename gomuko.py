import numpy as np
from copy import deepcopy
from state import Black, White

class Gomuko():
    def __init__(self, board_size):
        self.board = np.zeros(board_size)
        self.current_player = Black()
        self.board_size = board_size
        
    def restart(self):
        self.board = np.zeros_like(self.board)
        self.current_player = Black()
        
    def in_board(self, position):
        return (0 <= position[0] < self.board_size[0] 
                and 0 <= position[1] < self.board_size[1])
    @property
    def avilabel_moves(self):
        return np.where(self.board.flatten() == 0)[0]
    
    def move(self, position):
        position = (position // self.board_size[0], position % self.board_size[0])
        assert self.in_board(position)
        assert self.board[position] == 0
        
        self.board[position] = self.current_player.value
        result = self.is_end(position)
        if result != 0:
            return result, self.current_player
        self.current_player = self.current_player.oppent
        
        return result, None
        
    def is_end(self, position):
        index = np.where(self.board == 0)
        if len(index[0]) == 0:
            return -1
        
        directions = np.array([(-1,-1), (0,-1), (1,-1), (1,0), (1,1),
                               (0,1), (-1,1), (-1,0)])
        for direction in directions:
            count = 1
            for i in range(1,5):
                position_ = tuple(position + i*direction)
                if (not self.in_board(position_)
                    or self.board[position_] != self.current_player.value):
                    break
                    
                count += 1
                
            for i in range(-1, -5, -1):
                position_ = tuple(position + i*direction)
                if (not self.in_board(position_)
                    or self.board[position_] != self.current_player.value):
                    break
                    
                count += 1
                
            if count == 5:
                return 1

        return 0
    
    @property
    def features(self):
        black_features = np.zeros((self.board_size[0],self.board_size[1],1),dtype=np.float32)
        white_features = np.zeros((self.board_size[0],self.board_size[1],1),dtype=np.float32)
        player_features = np.ones((self.board_size[0],self.board_size[1],1),dtype=np.float32) * self.current_player.value
        
        black_stones_index = np.where(self.board == 1)
        white_stones_index = np.where(self.board == -1)
        
        black_features[black_stones_index] = 1
        white_features[white_stones_index] = 1
        
        return np.concatenate([black_features, white_features, player_features],2)
    
    def rate_model(self, black_player, white_player):
        self.restart()
        player_to_move = black_player
        while True:
            move, probs = player_to_move.player.get_action(self)
            result, winner = self.move(move)
            if result != 0:
                if result == 1:
                    if winner.name == "black":
                        winner_ = black_player
                        loser_ = white_player
                    else:
                        winner_ = white_player
                        loser_ = black_player
                    return winner_, loser_
                else:
                    return None, None
                break
            
            player_to_move = white_player if player_to_move.color == "black" else black_player
        
        return None, None
    
    def clone(self):
        return deepcopy(self)