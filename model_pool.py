import torch
import numpy as np
from policy_value_net import Model

class ModelPool:
    def __init__(self, layers, board_size, in_channels, device):
        self.pool = dict()
        self.layers = layers
        self.board_size = board_size
        self.in_channels = in_channels
        self.device = device
        
    def add_model(self, name, path, model, elo_rating=1500):
        self.pool[name] = [path, elo_rating]
        torch.save(model.net.state_dict(), path)
        
    def choice_model(self):
        model_name = np.random.choice(list(self.pool.keys()))
        path, elo_rating = self.pool[model_name]
        net = torch.load(path)
        model = Model(layers=self.layers, board_size=self.board_size,
                      in_channels=self.in_channels, device=self.device)
        model.load_param(path)
        return model_name, elo_rating, model
    
    def update_elo_rating(self, model_name, rating):
        self.pool[model_name][1] = rating