import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math
import os
import random
from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx
import powerlaw
from scipy.special import softmax
import itertools

from agents import Agent
from games import MujocoGame, ClevrGame

from egg.core.util import find_lengths


class Population(object):
    def __init__(self, population_size, n_pairs, world_name, population_name, learn_hps,
                 agent_hps, game_hps, world_hps, load_path=None):
        
        self.world_name = world_name
        self.load_path = load_path
        self.population_name = population_name
        self.population_size = population_size
        self.n_pairs = n_pairs

        self.agent_hps = agent_hps
        self.game_hps = game_hps
        self.learn_hps = learn_hps
        self.world_hps = world_hps

        self.data_path = self.game_hps["data_path"]
        self.n_games = self.game_hps["n_games"]
        self.game_type = os.path.basename(os.path.normpath(self.data_path))

        self.agents = []
        self.agent_steps = {}
        self.pairs = []
        self.network = None

        self.network_type = self.world_hps["network_type"]
        self.ordering = self.world_hps["ordering"]

        self.init_agents()
        
        # TODO: pass in appropriate network params         
        self.init_network(network_type=self.network_type,
                          n_nodes=population_size)  # construct networkx network of type `network_type`


        # collect edges from the graph and order according to some ordering algorithm
        self.compute_pairs()

        if load_path is not None:
            self.load_population()
            print("Loaded population from:", load_path)

        self.device = "cuda:0"

        self.summarize()

    def solve_poly(self, a, b, c):
        # ax**2 + bx + c = 0
        d = (b ** 2) - (4 * a * c)
        # return two solutions
        return (-b - math.sqrt(d)) / (2 * a), (-b + math.sqrt(d)) / (2 * a)

    def compute_barabasi_params(self, n_nodes, n_edges, verbose=False):
        solutions = self.solve_poly(1, n_nodes * -1.0, n_edges)
        m = int(min(solutions))
        if verbose:
            print(m, n_edges, solutions)
        return [n_nodes, m]

    def compute_erdos_params(self, n_nodes, n_edges):
        return [n_nodes, n_edges / ((n_nodes * n_nodes) / 2)]

    def compute_watts_params(self, n_nodes, n_edges, rewriting_prob=0.15):
        return [n_nodes, int((n_edges * 2) / n_nodes), rewriting_prob]

    def compute_network_params(self, network_type, n_nodes, n_edges):
        if network_type == "erdos":
            return self.compute_erdos_params(n_nodes, n_edges)
        elif network_type == "barabasi":
            return self.compute_barabasi_params(n_nodes, n_edges)
        elif network_type == "watts":
            return self.compute_watts_params(n_nodes, n_edges)

        return None
    
    def init_network(self, network_type, n_nodes, verbose=False):
        """
        construct networkx graphs
        """
        # TODO: should we pass n_edges in the constructor?
        n_edges = 100
        # get network settings
        network_settings = self.compute_network_params(network_type,
                                                       n_nodes,
                                                       n_edges)
        if network_type == "erdos":
            self.network = nx.erdos_renyi_graph(*network_settings)
        elif network_type == "barabasi":
            self.network = nx.barabasi_albert_graph(*network_settings)
        elif network_type == "watts":
            self.network = nx.connected_watts_strogatz_graph(*network_settings)
        
        else:  # construct random pairs
            s, r = [], []
            for i in range(n_nodes):
                for j in range(self.n_pairs):
                    s.append(i)
                    r.append(i)
            
            random.shuffle(s)
            random.shuffle(r)
            edges = list(zip(s, r))

            g = nx.MultiDiGraph()
            g.add_nodes_from(range(n_nodes))
            g.add_edges_from(edges)
            
            self.network = g

        if verbose:
            print("Generated network '{}' with {} nodes and {} edges".format(
                network_type, self.network.number_of_nodes(), self.network.number_of_edges()))

        return
    
    def compute_pairs(self):
        """
        sample pairs from the network constructed in `init_network` - controls the frequency with which agents appear

        set self.pairs to be a list of tuples of form (sender_id, receiver_id)
        where ids are integers in range(0, population_size)

        i.e. self.pairs = [(1, 9), (0, 2), ... , (4, 3)] 
        """
        edges = list(self.network.edges)
        
        if self.ordering == "random":
            edges = self.randomly_order(edges)
        
        else:
            pass
        
        self.pairs = edges

    def randomly_order(self, edges):
        """
        where `pairs` is a list
        """
        random.shuffle(edges)
        return edges
    
    def train(self):
        for pair in self.pairs:
            _ = self.train_single_pair(pair)
        
        self.save_population()
    
    def train_single_pair(self, pair):
        
        steps = 0
        all_rewards = []

        sender_id = pair[0]
        sender = self.agents[sender_id]
        sender.to(self.device)

        receiver_id = pair[1]
        receiver = self.agents[receiver_id]
        receiver.to(self.device)

        n_epochs = self.learn_hps["n_epochs"]
        batch_size = self.learn_hps["batch_size"]
        lr = self.learn_hps["lr"]

        print("Training on sender {:d} and receiver {:d}".format(sender_id, receiver_id))
        print("Sender has trained for {:d} steps".format(self.agent_steps[sender_id]))
        print("Receiver has trained for {:d} steps".format(self.agent_steps[receiver_id]))

        optimizer = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()), lr)

        for epoch in range(n_epochs):
            n_distractors = self.game_hps["n_distractors"]
            print("Training with {:d} distractors.".format(n_distractors))
            dataloader = self.init_dataloader(batch_size, n_distractors)

            bar = tqdm(dataloader)
            current_rewards = []

            for idx, game in enumerate(bar, start=1):
                optimizer.zero_grad()

                sender_input = game["target_img"].to(self.device)
                receiver_input = game["imgs"].to(self.device)
                labels = game["labels"].to(self.device)

                message, sender_logits, sender_entropy = sender(tgt_img=sender_input, mode="sender")
                choices, receiver_logits, receiver_entropy = receiver(message=message, img_set=receiver_input, mode="receiver")
                message_lengths = find_lengths(message)

                effective_sender_logits = self.mask_sender_logits(sender_logits, message_lengths)
                rewards = self.compute_rewards(choices, labels).detach()
                mean_rewards = rewards.mean().detach()
                current_rewards.append(mean_rewards)

                baseline = torch.mean(torch.Tensor(current_rewards)).detach()
                coeff = self.compute_entropy_coeff(steps, mean_rewards, baseline)
                loss = self.compute_loss(effective_sender_logits, receiver_logits, rewards, baseline, sender_entropy,
                                         coeff)

                if steps % 100 == 0:
                    current_mean = torch.mean(torch.Tensor(current_rewards))
                    bar.set_description("Mean Rewards: " + str(current_mean))
                    if eval:
                        all_rewards.append((steps, current_mean))
                    current_rewards = []

                loss.backward()
                optimizer.step()

                self.agent_steps[sender_id] += 1
                self.agent_steps[receiver_id] += 1
                steps += 1

        return all_rewards 
    
    def init_agents(self):
        for i in range(self.population_size):
            agent = Agent(self.agent_hps, self.game_type, self.compute_input_channels(self.game_type))
            self.agent_steps[i] = 0
            self.agents.append(agent)

    def init_dataloader(self, batch_size, n_distractors, eval=False):
        if self.game_type == "clevr":
            game = ClevrGame(self.data_path, num_distractors=n_distractors, num_games=self.n_games, eval=eval)
        elif self.game_type == "mujoco":
            game = MujocoGame(self.data_path, num_distractors=n_distractors, num_games=self.n_games, eval=eval)

        return DataLoader(game, batch_size=batch_size, shuffle=True)
    
    def compute_rewards(self, receiver_output, labels):
        rewards = (labels == receiver_output).float()

        return rewards
    
    def compute_loss(self, s_logits, r_logits, rewards, baseline, s_entropy, coeff):
        loss = (torch.sum(s_logits, 1) + r_logits) * -(rewards - baseline)

        entropy_term = coeff * s_entropy.mean().detach()

        return loss.mean() - entropy_term
    
    def compute_entropy_coeff(self, steps, rewards, baseline):
        if steps < 100000:
            coeff = 0.1 - torch.abs((rewards - baseline) * 0.1)
        else:
            coeff = 0.01
        
        return coeff
    
    def compute_input_channels(self, game_type):
        if game_type == "clevr":
            input_channels = 4
        elif game_type == "mujoco":
            input_channels = 3

        return input_channels
    
    def mask_sender_logits(self, sender_logits, message_lengths):
        effective_sender_logits = torch.zeros_like(sender_logits)

        message_length = self.agent_hps["message_length"]

        for i in range(message_length):
            not_eosed = (i <= message_lengths).float()
            effective_sender_logits[:, i] = sender_logits[:, i] * not_eosed

        return effective_sender_logits

    def load_population(self):
        for i in range(self.population_size):
            path = os.path.join("ecai_experiments", self.load_path, str(i))
            self.agents[i].load_state_dict(torch.load(path))
    
    def summarize(self):
        print("Initializing population of size {:d}.".format(self.population_size))
        print("Training {:d} pairs on the {:s} dataset.".format(len(self.pairs), self.game_type))
        print("Pairs are ordered {:s}.".format(self.ordering))
        print("\n")
        print("Pairs:", self.pairs)
        print("\n")
        print("Each agent will train for {:d} steps.".format(self.n_games * self.n_pairs))
        print("Partners per agent: {:d}".format(self.n_pairs))
        print("Epochs per pair: {:d}".format(self.learn_hps["n_epochs"]))
        print("Learning rate: {:f}".format(self.learn_hps["lr"]))
        print("Hidden size: {:d}".format(self.agent_hps["hidden_size"]))
        print("Embedding size: {:d}".format(self.agent_hps["emb_size"]))
        print("Vocab size: {:d}".format(self.agent_hps["vocab_size"]))
        print("Message length {:d}".format(self.agent_hps["message_length"]))
        print("\n")
