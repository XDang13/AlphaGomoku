import torch
import numpy as np
from torchvision.transforms import ToTensor
from collections import namedtuple

class Memory:
    def __init__(self, batch_size):
        self.buffer = []
        self.record = namedtuple("record", ["state", "log_prob", "mcts_prob", "score"])
        self.batch_size = batch_size
        self.transform = ToTensor()

    def reset(self):
        self.buffer = []
        
    def append(self, state, log_prob, mcts_prob, score):
        record = self.record(state, log_prob, mcts_prob, score)
        self.buffer.append(record)
        
    def hflip(self, record):
        state = np.copy(record.state)
        log_prob = np.copy(record.log_prob.reshape(state.shape[:2]))
        mcts_prob = np.copy(record.mcts_prob.reshape(state.shape[:2]))
        score = np.copy(record.score)
        
        state_ = np.flip(state, 1).copy()
        log_prob_ = np.flip(log_prob, 1).flatten().copy()
        mcts_prob_ = np.flip(mcts_prob, 1).flatten().copy()
        score_ = score.copy()
        
        return self.record(state_, log_prob_, mcts_prob_, score_)
    
    def vflip(self, record):
        state = np.copy(record.state)
        log_prob = np.copy(record.log_prob.reshape(state.shape[:2]))
        mcts_prob = np.copy(record.mcts_prob.reshape(state.shape[:2]))
        score = np.copy(record.score)
        
        state_ = np.flip(state, 0).copy()
        log_prob_ = np.flip(log_prob, 0).flatten().copy()
        mcts_prob_ = np.flip(mcts_prob, 0).flatten().copy()
        score_ = score.copy()
        
        return self.record(state_, log_prob_, mcts_prob_, score_)
    
    def dflip(self, record):
        state = np.copy(record.state)
        log_prob = np.copy(record.log_prob.reshape(state.shape[:2]))
        mcts_prob = np.copy(record.mcts_prob.reshape(state.shape[:2]))
        score = np.copy(record.score)
        
        state_ = np.flip(state, (0, 1)).copy()
        log_prob_ = np.flip(log_prob, (0,1)).flatten().copy()
        mcts_prob_ = np.flip(mcts_prob, (0, 1)).flatten().copy()
        score_ = score.copy()
        
        return self.record(state_, log_prob_, mcts_prob_, score_)
    
    def extend(self):
        buffer_h, buffer_v, buffer_d = [], [], []
        for record in self.buffer:
            record_h = self.hflip(record)
            record_v = self.vflip(record)
            record_d = self.dflip(record)
            
            buffer_h.append(record_h)
            buffer_v.append(record_v)
            buffer_d.append(record_d)
            
        self.buffer.extend(buffer_h)
        self.buffer.extend(buffer_v)
        self.buffer.extend(buffer_d)
        
    @property
    def sample(self):
        np.random.shuffle(self.buffer)
        transform = lambda x: self.transform(x).unsqueeze(0)
        from_numpy = lambda x: torch.from_numpy(x).unsqueeze(0)
        for i in range(0, len(self.buffer), self.batch_size):
            records = self.buffer[i: i+self.batch_size]
            states = [transform(record.state) for record in records]
            log_probs = [from_numpy(record.log_prob) for record in records]
            mcts_probs = [from_numpy(record.mcts_prob) for record in records]
            scores = [from_numpy(record.score) for record in records]
            
            states = torch.cat(states).float()
            log_probs = torch.cat(log_probs).float()
            mcts_probs = torch.cat(mcts_probs).float()
            scores = torch.cat(scores).float()
            
            yield states, log_probs, mcts_probs, scores