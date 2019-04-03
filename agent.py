import torch
import json
import numpy as np
import time
from policy_value_net import Model
from gomuko import Gomuko
from mcts import Player
from model_pool import ModelPool
from memory import Memory
#from vis import Vis
from collections import namedtuple, deque

class Agent:
    def __init__(self, board_size, pool_folder, batch_size=32, mcts_epoches=400, port=8097, device="cpu"):
        self.game = Gomuko(board_size)
        self.model = Model(layers=[2,2,2], board_size=board_size, in_channels=3, device=device)
        self.player = Player(self.model, mcts_epoches, True)
        self.memory = Memory(batch_size)
        self.pool = ModelPool(layers=[2,2,2], board_size=board_size, in_channels=3, device=device)
        #self.vis = Vis(port)
        self.loss_history = []
        self.index = 1
        self.generation = 0
        self.mcts_epoches = mcts_epoches
        self.pool_folder = pool_folder
        '''
        policy_loss_opts = {
            "title": "Policy Loss",
        }
        
        value_loss_opts = {
            "title": "Value Loss",
        }
        
        self.vis.add_line("policy_loss", 1, policy_loss_opts)
        time.sleep(0.001)
        self.vis.add_line("value_loss", 1, value_loss_opts)
        '''
    def collect_game_data(self):
        self.memory.reset()
        self.player.reset()
        self.game.restart()
        states = []
        log_probs = []
        mcts_probs = []
        current_player = []
        while True:
            move, probs = self.player.get_action(self.game)
            log_prob = self.model.get_log_prob(self.game)
            states.append(self.game.features)
            mcts_probs.append(probs)
            log_probs.append(log_prob)
            current_player.append(self.game.current_player.value)
            result, winner = self.game.move(move)
            if result != 0:
                break
        scores = np.zeros(len(current_player))
        if result != -1:
            scores[np.array(current_player) == winner.value] = 1.0
            scores[np.array(current_player) != winner.value] = -1.0
            
        for state, mcts_prob, score in zip(states, log_probs, mcts_probs, scores.tolist()):
            self.memory.append(state, log_prob, mcts_prob, np.array(score))
            
        self.memory.extend()
            
    def train(self):
        value_loss_history = []
        policy_loss_history = []
        index = 0
        for _ in range(5):
            for i, (states, log_probs, mcts_probs, scores) in enumerate(self.memory.sample):
                value_loss, policy_loss, loss = self.model.update(states, log_probs, mcts_probs, scores)
                policy_loss_history.append(policy_loss)
                value_loss_history.append(value_loss)
                self.loss_history.append([policy_loss, value_loss])
                
                mean_policy_loss = np.mean(policy_loss_history)
                mean_value_loss = np.mean(value_loss_history)
                
                
                iteration_str = "Epoch {} - Iteration {}".format(self.generation, index)
                policy_loss_str = "Average Policy Loss: {:.6f}".format(mean_policy_loss)
                value_loss_str = "Average Value Loss: {:.6f}".format(mean_value_loss)
                
                content = "{}: {} | {}".format(iteration_str, policy_loss_str, value_loss_str)
                
                print("\r{}".format(content), end="")    
                
                '''
                update_type = "new" if self.index == 5 else "append"

                self.vis.update_line("policy_loss", self.index, policy_loss, update_type)
                time.sleep(0.0001)
                self.vis.update_line("value_loss", self.index, value_loss, update_type)
                time.sleep(0.0001)
                '''
                index += 1
        
        mean_policy_loss = np.mean(policy_loss_history)
        mean_value_loss = np.mean(value_loss_history)

        iteration_str = "Epoch {} - Iteration {}".format(self.generation, index)
        policy_loss_str = "Average Policy Loss: {:.6f}".format(mean_policy_loss)
        value_loss_str = "Average Value Loss: {:.6f}".format(mean_value_loss)

        content = "{}: {} | {}".format(iteration_str, policy_loss_str, value_loss_str)

        print("\r{}".format(content))
                
    def store_model(self, name, path, elo_rating=1500):
        self.pool.add_model(name, path, self.model, elo_rating)
        
    def elo_rate(self, player_black_rate, player_white_rate, winner=None):
        ratio_black = (player_white_rate - player_black_rate) / 400
        ratio_white = (player_black_rate - player_white_rate) / 400
        estimate_black = 1 / (1 + 10 ** ratio_black)
        estimate_white = 1 / (1 + 10 ** ratio_white)
        
        if winner is None:
            s_black = 0.5
            s_white = 0.5
        else:
            s_black = 1 if winner == "black" else 0
            s_white = (1 - s_black)
        
        player_black_rate += 32 * (s_black - estimate_black)
        player_white_rate += 32 * (s_white - estimate_white)
        
        return round(player_black_rate), round(player_white_rate)
        
    def auto_play(self, init_rate):
        coin = np.random.random()
        competitor = namedtuple("Player",["name", "player", "elo", "color"])
        competitor_alpha_name = "G_" + str(self.generation)
        competitor_alpha_player = Player(self.model, self.mcts_epoches, False)
        competitor_alpha_rate = init_rate
        
        competitor_beta_name, competitor_beta_rate, competitor_beta_mode = self.pool.choice_model()
        competitor_beta_player = Player(competitor_beta_mode, self.mcts_epoches, False)
        if coin > 0.5:
            black_player = competitor(competitor_alpha_name, competitor_alpha_player,
                                      competitor_alpha_rate, "black")
            white_player = competitor(competitor_beta_name, competitor_beta_player,
                                      competitor_beta_rate, "white")
        else:
            black_player = competitor(competitor_beta_name, competitor_beta_player,
                                      competitor_beta_rate, "black")
            white_player = competitor(competitor_alpha_name, competitor_alpha_player,
                                      competitor_alpha_rate, "white")
            

        winner, loser = self.game.rate_model(black_player, white_player)
        
        if winner is not None:
            black_player_elo, white_player_elo = self.elo_rate(black_player.elo, 
                                                               white_player.elo, winner.color)
        else:
            black_player_elo, white_player_elo = self.elo_rate(black_player.elo,
                                                               white_player.elo)
            
        if coin > 0.5:
            competitor_alpha_rate = black_player_elo
            competitor_beta_rate = white_player_elo
        else:
            competitor_alpha_rate = white_player_elo
            competitor_beta_rate = black_player_elo
        
        self.pool.update_elo_rating(competitor_beta_name, competitor_beta_rate)
        
        return competitor_alpha_rate

    def evaluate(self, num=10):
        elo_rate = 1500
        for _ in range(num):
            elo_rate = self.auto_play(elo_rate)
        
        name = "Generation_{}".format(self.generation)
        #file = "model/{}".format(name)
        file = "{}{}.pt".format(self.pool_folder, name)
        self.store_model(name, file, elo_rate)
        
        print("\rGeneration | ELO")
        for key in self.pool.pool:
            elo = self.pool.pool[key][1]
            content = "{}: {}".format(key, elo)
            print(content)
            
        '''
        items = list(self.pool.pool.values())
        rownames = [item[0] for item in items]
        rates = [item[1] for item in items]
        opts = {
            "title": name,
            "rownames": rownames
        }
        x = torch.tensor(rates)
        self.vis.plot_bar("rate", x, opts)
        '''
        
    def fit(self, epoches, num=10):
        self.evaluate(0)
        for epoch in range(epoches):
            print("\rCollecting Data....", end="")
            self.generation += 1
            self.collect_game_data()
            self.train()
            if self.generation % 20 == 0:
                print("\rEvaluateing....", end="")
                self.evaluate(num)
            
        self.evaluate(num)
        
    def save_model(self, folder, name)

        print("Svaing Model")
        model_file = "{}/{}".format(folder, name)
        torch.save(self.model.net.state_dict(), model_file)
        print("Svaing info to Driver")
        pool_content = json.dumps(self.pool.pool)
        #json_file = "model/{}.json".format(name)
        pool_json_file = "{}/{}_pool.json".format(folder, name)
        with open(pool_json_file, 'w') as file:
            json.dump(pool_content, file)
            
        check_points = {
            "index": self.index,
            "generation": self.generation,
            "loss history": self.loss_history
        }
        
        check_points_json_content = json.dumps(check_points)
        check_points_json_file = "{}/{}_check_points.json".format(folder, name)
        with open(check_points_json_file, 'w') as file:
            json.dump(check_points_json_content, file)
        
    def load_model(self, model_file, pool_json_file, check_points_json_file):
        self.model.net.load_state_dict(torch.load(model_file))
        with open(pool_json_file, "r") as file:
            pool_json = json.load(file)
            
        self.pool.pool = json.loads(pool_json)
        
        with open(check_points_json_file, "r") as file:
            check_points_json = json.load(file)
        
        check_points = json.loads(check_points_json)
        self.loss_history = check_points["loss history"]
        self.generation = check_points["generation"]
        self.index = check_points["index"]