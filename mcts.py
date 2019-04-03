import math
import torch
import numpy as np

class Node():
    def __init__(self, parent=None, action=None, prob=1, coefficient=5):
        self.parent = parent
        self.action = action
        self.probs = prob
        self.children = []
        self.children_actions = []
        self.visits = 0
        self.Q = 0
        self.u_value = 0
        self.coefficient = coefficient
        
    @property
    def values(self):
        self.u_value = self.coefficient * self.probs * math.sqrt(self.parent.visits / (1+self.visits))
        
        return self.Q + self.u_value
    
    def select(self):
        
        node = max(self.children, key = lambda child: child.values)
        return node
    
    def add_child(self, actions_probs):
        for action, prob in actions_probs:
            if action not in self.children_actions:
                node = Node(self, action, prob.item())
                self.children.append(node)
                self.children_actions.append(action)
            
    def _update(self, value):
        self.visits += 1
        self.Q += (value - self.Q) / self.visits
        
    def update(self, value):
        if self.parent:
            self.parent.update(-value)
            
        self._update(value)
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent == None
        
class MCTS:
    def __init__(self, model, epoches, coefficient=5):
        self.model = model
        self.coefficient = coefficient
        self.epoches = epoches
        self.root = Node(coefficient=coefficient)
        
    def simulate(self, game):
        node = self.root
        result = 0
        winner = None
        while not node.is_leaf():
            node = node.select()
            result, winner = game.move(node.action)

        actions_probs, value = self.model.get_moves(game)
        if result == 0:
            node.add_child(actions_probs)
        else:
            if result == -1:
                value = 0
            else:
                value = 1 if winner == game.current_player else -1

        
        node.update(-value)

    def __call__(self, game, temp=1):
        for i in range(400):
            game_ = game.clone()
            self.simulate(game_)
                
        visits = [child.visits for child in self.root.children]

        probs = self.softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return self.root.children, probs

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs
    
    def update(self, node=None):
        if node is None:
            self.root = Node(coefficient=self.coefficient)
        else:
            self.root = node
            self.root.parent = None
            
class Player:
    def __init__(self, model, epoches, self_play):
        self.mcst = MCTS(model, epoches)
        self.self_play = self_play
        
    def reset(self):
        self.mcst.update()
    
    @torch.no_grad()
    def get_action(self, game):
        avilabel_moves = game.avilabel_moves
        mcts_probs = np.zeros(game.board_size).flatten()
        if len(avilabel_moves) > 0:
            nodes, probs = self.mcst(game)
            moves = [node.action for node in nodes]
            mcts_probs[moves] = probs
            if self.self_play:
                noise = np.random.dirichlet(np.ones_like(probs)*0.3) * 0.25
                probs = probs * 0.75 + noise
            node = np.random.choice(nodes, p=probs)
            if self.self_play:
                self.mcst.update(node)
            else:
                self.mcst.update(None)
                
            return node.action, mcts_probs
        else:
            print("Warning, no legal moves")