from pypokerengine.players import BasePokerPlayer
import numpy as np
import torch

from SimpleNet import SimpleNet

# Количсевто мастей
SUITS = 4

# Количество номиналов карт
CARD_VALUES = 13

CARD_SUITS = ['H', 'D', 'C', 'S']
CARDS_VALUES = [str(x) for x in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']
ACTIONS = ['fold', 'call', 'raise']


def card_to_one_hot_encode(card):
    suit = np.zeros(SUITS)
    value = np.zeros(CARD_VALUES)

    if card != 'unknown':
        suit[CARD_SUITS.index(card[0])] = 1
        value[CARDS_VALUES.index(card[1])] = 1

    return np.concatenate([suit, value])


def all_cards_encode(cards):
    all_cards = np.zeros((SUITS, CARD_VALUES))
    for card in cards:
        # Для карты в массиве карт выстовляем, что она есть.
        all_cards[CARD_SUITS.index(card[0]), CARDS_VALUES.index(card[1])] = 1
    return np.reshape(all_cards, SUITS * CARD_VALUES)


class NNPlayer(BasePokerPlayer):
    def __init__(self):
        self.model = SimpleNet()
        self.model.eval()

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']

        # Добавляем не открытые карты как неизвестные
        community_card_with_empty = list(community_card)
        if len(community_card_with_empty) < 5:
            community_card_with_empty += ['unknown'] * (5 - len(community_card_with_empty))

        self_card_encode = [card_to_one_hot_encode(x) for x in hole_card +
                            community_card_with_empty]
        self_card_encode = np.concatenate(self_card_encode)
        all_cards = all_cards_encode(hole_card + community_card)

        model_input = np.concatenate([self_card_encode, all_cards])
        model_input = torch.from_numpy(model_input).float()
        model_output = self.model(model_input)
        model_output = model_output.tolist()

        action = ACTIONS[model_output.index(max(model_output))]

        val_actions_list = [x['action'] for x in valid_actions]

        # Если действия выбранное моделью невозможно, то выбираем из возможных последнее
        if action not in val_actions_list:
            action = val_actions_list[-1]['action']

        if action == 'fold':
            amount = 0
        elif action == 'call':
            amount = valid_actions[1]["amount"]
        else:
            amount = valid_actions[2]["amount"]['min']

        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass