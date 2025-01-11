'''
6.9630 MIT POKERBOTS GAME ENGINE
DO NOT REMOVE, RENAME, OR EDIT THIS FILE
'''
from collections import namedtuple
from threading import Thread
from queue import Queue
import eval7
import sys
import os
import random

sys.path.append(os.getcwd())

from config import *

from engine import RoundState, FoldAction, CheckAction, CallAction, RaiseAction, TerminalState
from engine import STREET_NAMES,  CCARDS, PCARDS, PVALUE, STATUS
from training.local_player import LocalPlayer

class LocalGame():
    '''
    Manages logging and the high-level game procedure.
    '''

    def __init__(self, players):
        self.log = ['6.9630 MIT Pokerbots - ' + PLAYER_1_NAME + ' vs ' + PLAYER_2_NAME]
        self.players = players

    def log_round_state(self, players, round_state):
        '''
        Incorporates RoundState information into the game log and player messages.
        '''
        if round_state.street == 0 and round_state.button == 0:
            self.log.append('{} posts the blind of {}'.format(players[0].name, SMALL_BLIND))
            self.log.append('{} posts the blind of {}'.format(players[1].name, BIG_BLIND))
            self.log.append('{} dealt {}'.format(players[0].name, PCARDS(round_state.hands[0])))
            self.log.append('{} dealt {}'.format(players[1].name, PCARDS(round_state.hands[1])))
        elif round_state.street > 0 and round_state.button == 1:
            board = round_state.deck.peek(round_state.street)
            self.log.append(STREET_NAMES[round_state.street - 3] + ' ' + PCARDS(board) +
                            PVALUE(players[0].name, STARTING_STACK-round_state.stacks[0]) +
                            PVALUE(players[1].name, STARTING_STACK-round_state.stacks[1]))
            self.log.append(f"Current stacks: {round_state.stacks[0]}, {round_state.stacks[1]}")

    def log_action(self, name, action, bet_override):
        '''
        Incorporates action information into the game log and player messages.
        '''

        if isinstance(action, FoldAction):
            phrasing = ' folds'
        elif isinstance(action, CallAction):
            phrasing = ' calls'
        elif isinstance(action, CheckAction):
            phrasing = ' checks'
        elif isinstance(action, RaiseAction):
            phrasing = (' bets ' if bet_override else ' raises to ') + str(action.amount)

        self.log.append(name + phrasing)

    def log_terminal_state(self, players, round_state):
        '''
        Incorporates TerminalState information into the game log and player messages.
        '''
        previous_state = round_state.previous_state
        if FoldAction not in previous_state.legal_actions():
            self.log.append('{} shows {}'.format(players[0].name, PCARDS(previous_state.hands[0])))
            self.log.append('{} shows {}'.format(players[1].name, PCARDS(previous_state.hands[1])))
        self.log.append('{} awarded {}'.format(players[0].name, round_state.deltas[0]))
        self.log.append('{} awarded {}'.format(players[1].name, round_state.deltas[1]))
        hit_chars = ['0', '0']
        if round_state.bounty_hits[0]:
            hit_chars[0] = '1'
        if round_state.bounty_hits[1]:
            hit_chars[1] = '1'
        if round_state.deltas[0] > 0: # mask out the losing player's hit
            hit_chars[1] = '#'
        elif round_state.deltas[1] > 0:
            hit_chars[0] = '#'
    
    def query(self, round_state, game_log):
        '''
        Requests one action from the pokerbot over the socket connection.

        This method handles communication with the bot, sending the current game state
        and receiving the bot's chosen action. It enforces game clock constraints and
        validates that the received action is legal.

        Args:
            round_state (RoundState or TerminalState): The current state of the game.
            player_message (list): Messages to be sent to the player bot, including game state
                information like time remaining, player position, and cards.
            game_log (list): A list to store game events and error messages.

        Returns:
            Action: One of FoldAction, CallAction, CheckAction, or RaiseAction representing
            the bot's chosen action. If the bot fails to provide a valid action, returns:
                - CheckAction if it's a legal move
                - FoldAction if check is not legal

        Notes:
            - The game clock is decremented by the time taken to receive a response
            - Invalid or illegal actions are logged but not executed
            - Bot disconnections or timeouts result in game clock being set to 0
            - At the end of a round, only CheckAction is considered legal
        '''
        legal_actions = round_state.legal_actions() if isinstance(round_state, RoundState) else {CheckAction}
        active = round_state.button%2
        player = self.players[active]
        action = player.get_action(round_state, active) #NEED TO HAVE WAY TO TRACK BETS
            
        if action in legal_actions:
            if isinstance(action, RaiseAction):
                amount = action.amount
                min_raise, max_raise = round_state.raise_bounds()
                if min_raise <= amount <= max_raise:
                    return action(amount)
            else:
                return action()
            
        game_log.append(self.name + ' attempted illegal ' + action.__name__)
        return CheckAction() if CheckAction in legal_actions else FoldAction()

    def run_round(self, players, bounties):
        '''
        Runs one round of poker.
        '''

        for player in players:
            player.handle_new_round()


        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        round_state = RoundState(0, 0, pips, stacks, hands, deck, bounties, None)
        while not isinstance(round_state, TerminalState):
            self.log_round_state(players, round_state)
            active = round_state.button % 2
            player = players[active]
            action = player.get_action(round_state, self.log, active)
            bet_override = (round_state.pips == [0, 0])
            self.log_action(player.name, action, bet_override)
            round_state = round_state.proceed(action)
        self.log_terminal_state(players, round_state)
        
        for player,delta in zip(players,round_state.deltas):
            player.bankroll += delta


    def log_round_states(self, players, bounties):
        '''
        Runs one round of poker.
        '''

        round_state_log = []

        for player in players:
            player.handle_new_round()

        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        round_state = RoundState(0, 0, pips, stacks, hands, deck, bounties, None)
        while not isinstance(round_state, TerminalState):
            self.log_round_state(players, round_state)
            active = round_state.button % 2
            player = players[active]
            action = player.get_action(round_state, self.log, active)
            bet_override = (round_state.pips == [0, 0])
            self.log_action(player.name, action, bet_override)

            round_state_log.append(round_state)
            round_state = round_state.proceed(action)
        self.log_terminal_state(players, round_state)
        
        return round_state_log
    
    def simulate_round_state(self, players, roundstate):

        roundstate.deck.shuffle()

        while not isinstance(round_state, TerminalState):
            self.log_round_state(players, round_state)
            active = round_state.button % 2
            player = players[active]
            action = player.get_action(round_state, self.log, active)
            bet_override = (round_state.pips == [0, 0])
            self.log_action(player.name, action, bet_override)

            round_state.append(round_state)
            round_state = round_state.proceed(action)
        self.log_terminal_state(players, round_state)
        
        return round_state.deltas
    
    
    def run(self):
        '''
        Runs one game of poker.
        '''
        print('   __  _____________  ___       __           __        __    ')
        print('  /  |/  /  _/_  __/ / _ \\___  / /_____ ____/ /  ___  / /____')
        print(' / /|_/ // /  / /   / ___/ _ \\/  \'_/ -_) __/ _ \\/ _ \\/ __(_-<')
        print('/_/  /_/___/ /_/   /_/   \\___/_/\\_\\\\__/_/ /_.__/\\___/\\__/___/')
        print()
        print('Starting the Pokerbots engine...')

        players = self.players

        for round_num in range(1, NUM_ROUNDS + 1):
            self.log.append('')
            self.log.append('Round #' + str(round_num) + STATUS(players))
            if round_num % ROUNDS_PER_BOUNTY == 1:
                cardNames = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                bounties = [cardNames[random.randint(0, 12)], cardNames[random.randint(0, 12)]]
                self.log.append(f"Bounties reset to {bounties[0]} for player {players[0].name} and {bounties[1]} for player {players[1].name}")
            self.run_round(players, bounties)
            self.log.append('Winning counts at the end of the round: ' + STATUS(players))

            players = players[::-1]
            bounties = bounties[::-1]
        self.log.append('')
        self.log.append('Final' + STATUS(players))
        # for player in players:
        #     player.stop()
        name = GAME_LOG_FILENAME + '.txt'
        print('Writing', name)
        with open(name, 'w') as log_file:
            log_file.write('\n'.join(self.log))


if __name__ == '__main__':

    player_0 = LocalPlayer("Player A")
    player_1 = LocalPlayer("Player B")

    players = [player_0, player_1]
    LocalGame(players).run()

        





