import torch
import visdom

class Vis:
    def __init__(self, port=8097):
        self.vis = visdom.Visdom(port=port)
        self.windows = dict()
    
    def add_line(self, name, num_lines, opts):
        win = self.vis.line(
            X = torch.ones((1,)),
            Y = torch.zeros((1, num_lines)),
            opts = opts
        )
        
        self.windows[name] = win
   
        
    def update_line(self, name, x, y, update_type):
        self.vis.line(
            X = torch.ones((1,)) + x,
            Y = torch.tensor([y]),
            win = self.windows[name],
            update = update_type
        )
        
    def plot_bar(self, name, values, opts):
        if name not in self.windows:
            win = self.vis.bar(
                X = values,
                opts = opts
            )
        else:
            self.vis.bar(
                X = values,
                opts = opts,
                win = self.windows[name]
            )