import random
import math
import torch 

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import *

from local_engine import LocalGame
from local_player import LocalPlayer
from neural_net import DeepCFRModel
from local_player import LocalPlayer
from engine import FoldAction, CheckAction, CallAction, RaiseAction, TerminalState

#TODO: Modify the roundstate class such that it also keeps a bet history

class MemoryReservoir:

    def __init__(self, capacity):
        """
        Initialize the MemoryReservoir with a fixed capacity.
        
        :param capacity: The maximum number of items the reservoir can hold.
        """
        self.capacity = capacity
        self.memory = []
        self.stream_size = 0

    def add(self, item):
        """
        Add a new item to the reservoir using reservoir sampling.
        
        :param item: The item to be added.
        """
        self.stream_size += 1
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            replace_index = random.randint(0, self.stream_size - 1)
            if replace_index < self.capacity:
                self.memory[replace_index] = item
    
    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)
    


class CFR:

    def __init__(self, cfr_iters, mcc_iters, round_iters, reservouir_size):

        self.cfr_iters = cfr_iters
        self.mcc_iters = mcc_iters
        self.round_iters = round_iters

        self.num_epochs = 10
        self.batch_size = 100
        self.learning_rate = 0.1

        self.game_engine = LocalGame()

        self.nbets = 8
        self.nactions = 5

        self.roundstate_memory = [MemoryReservoir(reservouir_size) for i in [0,1]]
        self.advantage_memory = [MemoryReservoir(reservouir_size) for i in [0,1]]
        self.strategy_memory = [[], []]
        self.models = [None, None]

    def generate_model(self):

        return DeepCFRModel(4, self.nbets, self.nactions)
    
    def generate_roundstates(self, traverser):

        t = random.randint(0,len(self.strategy_memory[0])-1)

        if traverser == 0:
            player_A = LocalPlayer("A", self.strategy_memory[0][-1])
            player_B = LocalPlayer("B", self.strategy_memory[1][t])

        elif traverser == 1:
            player_A = LocalPlayer("A", self.strategy_memory[0][t])
            player_B = LocalPlayer("B", self.strategy_memory[1][-1])

        players = [player_A, player_B]

        bounties = []

        roundstates = self.game_engine.generate_roundstates(players, bounties)

        for roundstate in roundstates:

            active = roundstate.button%2

            self.roundstate_memory[active].add((roundstate, active, t))

    def infoset_EV(self, roundstate, traverser, t):

        if traverser == 0:
            player_A = LocalPlayer("A", self.strategy_memory[0][-1])
            player_B = LocalPlayer("B", self.strategy_memory[1][t])

        elif traverser == 1:
            player_A = LocalPlayer("A", self.strategy_memory[0][t])
            player_B = LocalPlayer("B", self.strategy_memory[1][-1])

        players = [player_A, player_B]
        
        pnls = torch.zeros(2)

        for i in range(self.mcc_iters):

            round_pnls = self.game_engine.simulate_round_state(players, roundstate)
            round_pnls = torch.tensor(round_pnls)
            pnls += round_pnls

        pnls = pnls / self.mcc_iters

        return pnls[traverser]
    
    def action_EV(self, roundstate, traverser, t, action):

        active = roundstate.button%2

        if traverser != active:
            raise ValueError("Can not estimate action values for non active player")

        new_roundstate = roundstate.proceed(action)

        pnls = self.infoset_EV(new_roundstate, traverser, t)
        
        return pnls
    
    def regret_tensor(self, roundstate, traverser, t):

        regret_tensor = torch.zeros(5)
        legal_mask = self.models[traverser].tensorize_mask(roundstate)

        pot = sum([STARTING_STACK - roundstate.stacks[i] for i in [0,1]])

        half_pot = RaiseAction(math.ceil(pot*1/2))
        three_half_pot = RaiseAction(math.ceil(pot*1/2))

        action_space = [FoldAction(), CheckAction(), CallAction(), half_pot, three_half_pot]

        infoset_EV = self.infoset_EV(roundstate, traverser, t)

        for action_idx, action in enumerate(action_space):

            if legal_mask[action_idx] > 0.001:

                action_EV = self.action_EV(roundstate, traverser, t, action)
                regret_tensor[action_idx] = action_EV - infoset_EV


        return regret_tensor.unsqueeze(0)
    
    def training_data(self, traverser):

        data = []

        for i in range(self.round_iters):

            self.generate_roundstates(traverser)
        
        for info in self.roundstate_memory[traverser]:

            roundstate, traverser, t = info

            regret_tensor = self.regret_tensor(roundstate, traverser, t)

            card_tensor_list, bet_tensor = self.models[traverser].tensorize_roundstate(roundstate, traverser)

            data.append((card_tensor_list, bet_tensor, regret_tensor))


        return data

    def batchify(self, data, batch_size):

        inputs_bets = []
        inputs_cards = [[] for _ in range(4)]
        regrets = []
    
        for (cards, bets, regret_vec) in data:

            inputs_bets.append(bets)
            for j in range(4):
                inputs_cards[j].append(cards[j])
            regrets.append(regret_vec)
        
        # When we have a full batch, yield the batch
        if len(inputs_bets) == batch_size:
            batched_bets = torch.cat(inputs_bets, dim=0)
            batched_cards = [torch.cat(inputs_cards[j], dim=0) for j in range(4)]
            batched_regrets = torch.tensor(regrets, torch.float32)
            
            yield (batched_bets, batched_cards), batched_regrets
            
            # Reset for the next batch
            inputs_bets = []
            inputs_cards = [[] for _ in range(4)]
            regrets = []
    
        # Handle remaining data if any
        if inputs_bets:
            batched_bets = torch.cat(inputs_bets, dim=0)
            batched_cards = [torch.cat(inputs_cards[j], dim=0) for j in range(4)]
            batched_regrets = torch.cat(regrets, dim= 0)
            yield (batched_cards, batched_bets), batched_regrets

    def train_model(self, traverser):

        new_model = self.generate_model()
        self.models[traverser] = new_model
        training_data = self.training_data(traverser)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):

            epoch_loss = 0.0
            optimizer.zero_grad()

            for (batched_cards, batched_bets), mcc_regrets in self.batchify(training_data, self.batch_size):

                predicted_regrets = new_model(batched_cards, batched_bets)

                loss = criterion(predicted_regrets.squeeze(), mcc_regrets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()


            print("---------------------")
            print("Training model number ", len(self.strategy_memory[0]) + 1)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

        print("What is the class of the output in train model: ", type(new_model))
        print("Training complete!")
        return new_model
    
    def run_CFR(self):

        new_models = []

        for traverser in [0,1]:

            new_models.append(self.train_model(traverser))

        for i in [0,1]:

            self.advantage_memory.append(self.models[i])

            self.models[i] = new_models[i]

    def CFR_training(self):

        for i in [0,1]:
            rand_model = self.generate_model()
            self.strategy_memory[i].append(rand_model)
        
        for i in range(self.cfr_iters):
            self.run_CFR()
        

            



















    
            

        






