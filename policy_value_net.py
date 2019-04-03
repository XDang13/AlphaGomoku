import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from basebone import ResNetBasebone, BasicBlock, BasicConv2d

class PolicyValueNet(ResNetBasebone):
    def __init__(self, block, layers, in_channels, board_size, active_fn=nn.SELU()):
        super(PolicyValueNet, self).__init__()
        self.in_channels = 32
        self.active_fn = active_fn
        self.block_1 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, bn=True,
                        active_fn=active_fn, stride=1, padding=1, bias=False),
        )
        
        self.block_2 = self._make_layer(block, 32, layers[0])
        self.block_3 = self._make_layer(block, 64, layers[1])
        self.block_4 = self._make_layer(block, 128, layers[2])
        self.block_fc = nn.Sequential(
            nn.Linear(board_size[0] * board_size[1] * 128, 1024),
            self.active_fn,
        )
        
        
        self.policy_block = nn.Sequential(
            nn.Linear(1024, 1024),
            self.active_fn,
            nn.Linear(1024, board_size[0] * board_size[1]),
        )
        
        self.value_block = nn.Sequential(
            nn.Linear(1024, 1024),
            self.active_fn,
            nn.Linear(1024, 1),
        )
        
    def forward(self, x):
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        output = output.view(output.size(0),-1)
        output = self.block_fc(output)
        action_probs = self.policy_block(output)
        action_probs = F.log_softmax(action_probs,dim=1)
        value = self.value_block(output)
        return action_probs, value
    
    def log_prob(self, x):
        output = self.block_1(x)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        output = output.view(output.size(0),-1)
        output = self.block_fc(output)
        action_probs = self.policy_block(output)
        action_probs = F.log_softmax(action_probs,dim=1)
        
        return action_probs
    
def build_net(**kwargs):
    return PolicyValueNet(BasicBlock, **kwargs)

class Model:
    def __init__(self, layers=[2,2,2], board_size=9, in_channels=3,
                 lr=1e-4, epsilon = 0.2, device="cpu"):
        self.device = torch.device(device)
        self.net = build_net(layers=layers, board_size=board_size,
                             in_channels=in_channels).to(self.device)
        self.transform = ToTensor()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.epsilon = epsilon
    
    @torch.no_grad()
    def get_log_prob(self, board):
        features = self.transform(board.features).unsqueeze(0).to(self.device)
        log_action_probs = self.net.log_prob(features)
        
        log_probs = log_action_probs.cpu().detach().numpy().flatten()
        
        return log_probs
        
    def get_moves(self, board):
        features = self.transform(board.features).unsqueeze(0).to(self.device)
        avilabel_moves = board.avilabel_moves
        log_action_probs, value = self.net(features)
        actions = torch.exp(log_action_probs)
        actions = actions.cpu().detach().numpy().flatten()
        return zip(avilabel_moves, actions[avilabel_moves]), value
    
    def load_param(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update(self, states, log_probs, mcts_probs, scores):
        states = states.to(self.device)
        log_probs = log_probs.to(self.device)
        mcts_probs = mcts_probs.to(self.device)
        scores = scores.to(self.device)
        
        pred_log_probs, pred_values = self.net(states)
        
        value_loss = F.mse_loss(pred_values, scores)
        
        ratio = torch.exp(pred_log_probs - log_probs)
        surr_1 = ratio * mcts_probs
        surr_2 = ratio.clamp(1-self.epsilon, 1+self.epsilon) * mcts_probs
        
        policy_loss = -(torch.min(surr_1, surr_2).sum(dim=1).mean())
       
        loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return value_loss.item(), policy_loss.item(), loss.item()#S, kl_dist
        