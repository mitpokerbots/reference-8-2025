import random
import math

import torch as t

from training.local_engine import LocalGame
from training.local_player import LocalPlayer
from training.neural_net import DeepCFRModel
from training.local_player import LocalPlayer
from engine import RoundState, FoldAction, CheckAction, CallAction, RaiseAction, TerminalState

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

class CFR:

    def __init__(self, cfr_iters, mcc_iters, round_iters, reservouir_size):

        self.cfr_iters = cfr_iters
        self.mcc_iters = mcc_iters
        self.round_iters = round_iters

        self.game_engine = LocalGame()

        self.roundstate_memory = [MemoryReservoir(reservouir_size) for i in [0,1]]
        self.advantage_memory = [MemoryReservoir(reservouir_size) for i in [0,1]]
        self.strategy_memory = [[], []]
        self.models = [None, None]

    def generate_model(self):

        return DeepCFRModel(4, self.nbets, self.nactions)
    
    def generate_roundstates(self, traverser):

        t = random.randint(0,len(self.advantage_memory)-1)

        if traverser == 0:
            player_A = LocalPlayer("A", self.strategy_memory[-1])
            player_B = LocalPlayer("B", self.strategy_memory[t])

        elif traverser == 1:
            player_A = LocalPlayer("A", self.strategy_memory[t])
            player_B = LocalPlayer("B", self.strategy_memory[-1])

        players = [player_A, player_B]

        roundstates = self.game_engine.log_round_state(players)

        for roundstate in roundstates:

            active = roundstate.button%2

            self.round_state_memory[active].add((roundstate, active, t))
        
    def tensorize_legal_mask(self, roundstate):

        legal_actions = roundstate.legal_actions()

        mask_tensor = t.zeros(self.nactions, dtype=t.float32)

        if FoldAction in legal_actions:
            mask_tensor[0] = 1
        
        if CheckAction in legal_actions:
            mask_tensor[1] = 1
        
        if CallAction in legal_actions:
            mask_tensor[2] = 1
        
        if RaiseAction in legal_actions:

            if self.min_raise <= math.ceil(self.pot*1/2) <= self.max_raise:
                mask_tensor[3] = 1
            if self.min_raise <= math.ceil(self.pot*3/2) <= self.max_raise:
                mask_tensor[4] = 1
        
        return mask_tensor
    
    def infoset_EV(self, roundstate, traverser, t):

        if traverser == 0:
            player_A = LocalPlayer("A", self.strategy_memory[-1])
            player_B = LocalPlayer("B", self.strategy_memory[t])

        elif traverser == 1:
            player_A = LocalPlayer("A", self.strategy_memory[t])
            player_B = LocalPlayer("B", self.strategy_memory[-1])
        
        players = [player_A, player_B]
        
        pnls = t.zeros(2)

        for i in range(self.mcc_iters):

            round_pnls = self.game_engine.simulate_round_state(roundstate, players)
            round_pnls = t.tensor(round_pnls)
            pnls += round_pnls

        pnls = pnls / self.mcc_iters

        return pnls[traverser]
    
    def action_EV(self, roundstate, traverser, t, action):

        if traverser != roundstate.active:
            raise ValueError("Can not estimate action values for non active player")

        new_roundstate = roundstate.proceed(action)

        pnls = self.infoset_EV(new_roundstate, traverser, t)
        
        return pnls
    
    def regret_tensor(self, roundstate, traverser, t):

        regret_tensor = t.zeros(5)
        legal_mask = self.tensorize_legal_mask(roundstate)
        half_pot = RaiseAction(math.ceil(roundstate.pot*1/2))
        three_half_pot = RaiseAction(math.ceil(roundstate.pot*1/2))
        action_space = [FoldAction, CheckAction, CallAction, half_pot, three_half_pot]

        infoset_EV = self.infoset_EV(roundstate, traverser, t)

        for action_idx, action in enumerate(action_space):

            if legal_mask[action_idx] > 0.001:

                action_EV = self.action_EV(roundstate, traverser, t, action)
                regret_tensor[action_idx] = action_EV - infoset_EV


        return roundstate
    
    def training_data(self, traverser):

        input_data = []
        output_data = []

        for i in range(self.round_states):
            self.generate_roundstates()
        
        for info in self.roundstate_memory[traverser]:

            roundstate, traverser, t = info

            regret_tensor = self.regret_tensor(roundstate, traverser, t)

            input_data.append(roundstate)
            output_data.append(regret_tensor) 

        return (input_data, output_data)


#TODO: Actually make it so that when we get the input/output data we just store 
# The data in the following set up: (input_tensors, regret_tensor, legal_mask)
# THis will allow us to train using batching and so on and make things much faster
    
    def train_CFR_iter(self):



        for traverser in [0,1]:

            input_data, output_data = self.training_data(traverser)
















    
            

        






