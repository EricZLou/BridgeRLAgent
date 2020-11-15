"""
CS 238 Final Project: Bridge RL Agent
Eric Lou & Kimberly Tran
"""
import copy
import datetime
import numpy as np
import random

from collections import namedtuple


"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
REPRESENTATIONS OF BRIDGE

Representing a "Card" as an integer:
  	Cards 0 -> 12 are Club 2 -> Club 14
  	Cards 13 -> 25 are Diamond 2 -> Diamond 14
  	Cards 26 -> 38 are Heart 2 -> Heart 14
  	Cards 39 -> 51 are Spade 2 -> Spade 14

	Jack is 11
	Queen is 12
	King is 13
	Ace is 14

Representing a "Suit" as an integer:
	n/a is -1 <-- used in a "State" where no cards have been played yet.
	Clubs is 0
	Diamonds is 1
	Hearts is 2
	Spades is 3

Representing a "State" as an opening suit and frozenset of up to 3 "Card"-s:
    state = State(1, frozenset(23, 0))
	We have a Diamond 12 and Club 2 with an opening suit of Diamonds. 
	The agent is 3rd to play a card and must play a Diamond if it has one.

Representing the MDP with a Map from a "State" to an array of length-52:
	We call this Map "weights". And the array of length-52 represets the 
	proportion with which the agent should play each of the 52 cards given
	that it is at that state.
	In this example, with state = (1, set(23, 0)), weights[state] will 
	likely have very large values at indices 24 and 25 since a 
	Diamond 13 and Diamond 14 will beat the Diamond 12.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
State = namedtuple('State', ['opening_suit', 'cards_played', 'partners_card'])


"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"
" DEFINE SOME CONSTANTS
"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
NUM_ACTIONS = 52  # Agent can choose any card to play (only some are valid).
NUM_GAMES_TRAIN = 10000
NUM_GAMES_TEST = 10000
STATS_PER = 1000


"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"
" RL AGENT
"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
class BridgeAgent:
    def __init__(self):
        # We initialize all weights to 1 such that every card has an equal chance of being chosen.
        self.weights = {}

        self.weights[State(-1, frozenset(), -1)] = np.full(NUM_ACTIONS, 1.0)
        for opening_suit in range(4):
            for card_1 in range(52):
                for card_2 in range(card_1, 52):
                    for card_3 in range(card_2, 52):
                        for card_partner in [-1, card_1, card_2, card_3]:
                            state = State(
                                opening_suit, 
                                frozenset([card_1, card_2, card_3]), 
                                card_partner)
                            self.weights[state] = np.full(NUM_ACTIONS, 1.0)

        # self.alpha = 0.997      # 1,000
        # self.alpha = 0.9995     # 10,000
        # self.alpha = 0.99995    # 100,000
        self.alpha = 0.999995   # 1,000,000
        # self.alpha = 0.9999995  # 5,000,000
        self.game_num = 1

    """
	EXAMPLE
	state = State(1, set(23, 0))	# Diamond 12, Club 2		<-- first 2 cards in round
	card_played = 24				# Diamond 13				<-- 3rd card in round

	If 4th card is not 25, then the agent wins. We want to incrase the proportion
	with which we play 24.

	ba.add_win(state, card_played)
	"""
    def add_win(self, state, card_played):
        self.weights[state][card_played] *= (1 + 0.1 * self.alpha ** self.game_num)

    """
	EXAMPLE
	state = State(1, set(23, 0))
	card_played = 24

	If 4th card is 25 (Diamond 14), then the agent loses. We want to decrease the
	proportion with which we play 24.

	ba.add_loss(state, card_played)
	"""
    def add_loss(self, state, card_played):
        self.weights[state][card_played] /= (1 + 0.1 * self.alpha ** self.game_num)

    """
	EXAMPLE
	state = State(1, set(23, 0))
	cards_in_hand = set(0, 1, 4, 8, 11, 20, 24, 38)

	The agent choose to play whichever remaining card has the highest weight.
	The agent must play a Diamond if it has Diamonds. In this example, agent 
	will most likely play 24, which beats 23 <-- hopefully 24 has the highest
	weight.

	card_played = ba.play_card(state, cards_in_hand)
	"""
    def play_card(self, state, cards_in_hand):
        # Following the EXAMPLE:
        # suit = 1
        suit = state.opening_suit
        # valid_cards = [20, 24]
        valid_cards = np.array([i for i in range(suit * 13, (suit + 1) * 13) if i in cards_in_hand])

        if len(valid_cards) == 0:
            valid_cards = cards_in_hand

        # Choose the valid card with highest weight.
        # index_into_valid_counts = 1 since 20 has a smaller weight than 24.
        # index_into_valid_cards = np.argmax(self.weights[state][valid_cards])
        index_into_valid_cards = np.random.choice(np.flatnonzero(self.weights[state][valid_cards] == self.weights[state][valid_cards].max()))
        # returns valid_cards[1] = 24
        return valid_cards[index_into_valid_cards]

    """
    This function write the policy at the end of the data training phase.
    """
    def write_policy(self, cards_in_hand, policy, filename, states_accessed):
        count = 0
        with open(filename + "_Last_Game.txt", 'w') as g:
            g.write("Cards in Hand: {}\n\n".format(cards_in_hand))
            with open(filename + ".txt", 'w') as f:
                for state in self.weights:
                    f.write("State: suit {} | cards played {} | partner's card {}\nBest Card To Play: {}\n\n".format(state.opening_suit,
                                                                                                 state.cards_played, state.partners_card,
                                                                                                 policy[count]))
                    if state in states_accessed:
                        g.write("State: suit {} | cards played {} | partner's card {}\nBest Card To Play: {}\n\n".format(state.opening_suit,
                                                                                                     state.cards_played, state.partners_card,
                                                                                                     policy[count]))
                    count += 1



"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"
" UTILITY FUNCTIONS
"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
"""
This functions deals random cards.
"""
deck = list(range(52))
def shuffle_cards():
    random.shuffle(deck)
    return [deck[0:13], deck[13:26], deck[26:39], deck[39:52]]

"""
This function is used by non-agents who play randomly.
"""
def play_random_card(suit, cards_in_hand):
    if suit == -1:
        return random.choice(cards_in_hand)

    valid_cards = [i for i in range(suit * 13, (suit + 1) * 13) if i in cards_in_hand]

    if len(valid_cards) == 0:
        return random.choice(cards_in_hand)

    return random.choice(valid_cards)

"""
This function determines the winner of the round.
"""
def determine_round_winner(suit, cards_played):
    max_idx = -1
    max_val = -1
    for idx, card in enumerate(cards_played):
        if suit * 13 <= card < (suit + 1) * 13 and card > max_val:
            max_val, max_idx = card, idx
    return max_idx

"""
This function determines the declarer based on partnership with the most points.
Return: (agent_is_declarer, declarer_idx)
"""
def agent_declarer(hands):
    points = count_points(hands) # determines the number of points in each hand

    # agent's partnership has more points and agent is declarer
    if points[0] + points[2] > points[1] + points[3] and points[2] > points[0]:
        return True, 2

    # agent is not declarer and agent should start the play
    return False, -1

"""
This function counts the points in each hand.
Note: Ace is 12, 25, 38, 51
"""
def count_points(hands):
    points = []
    for hand in hands:
        p = 0
        for card in hand:
            if card % 13 == 12:
                p += 4
            elif card % 13 == 11:
                p += 3
            elif card % 13 == 10:
                p += 2
            elif card % 13 == 9:
                p += 1
        points.append(p)
    return points


"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"
" TRACKS PERFORMANCE OF BRIDGE AGENT
"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
class BridgeAgentRedFlags:
    def __init__(self):
        self.RED_FLAG_VIOLATIONS = np.zeros(3)
        self.RED_FLAG_TOTAL_COUNT = np.zeros(3)
        self.ALL_RED_FLAG_VIOLATIONS = np.zeros(3)      # Cumulative
        self.ALL_RED_FLAG_TOTAL_COUNT = np.zeros(3)     # Cumulative

    def clear_red_flags(self):
        self.RED_FLAG_VIOLATIONS = np.zeros(3)
        self.RED_FLAG_TOTAL_COUNT = np.zeros(3)

    """
    This function checks if the agent plays their highest card even though the 
    highest card already played is higher than theirs.
    """
    def highest_card(self, valid_cards, agent_valid_cards, card):
        if len(agent_valid_cards) > 1 and max(valid_cards) > max(agent_valid_cards):
            self.RED_FLAG_TOTAL_COUNT[0] += 1
            self.ALL_RED_FLAG_TOTAL_COUNT[0] += 1
            if card == max(agent_valid_cards):
                self.RED_FLAG_VIOLATIONS[0] += 1
                self.ALL_RED_FLAG_VIOLATIONS[0] += 1

    """
    This function checks if the agent wins a round when there's three cards played already 
    and the agent has at least one higher card than what's been played.
    """
    def higher_card(self, valid_cards, agent_valid_cards, card, cards_played, partners_cards):
        if (len(cards_played) == 3 and len(agent_valid_cards) > 1 and 
            max(agent_valid_cards) > max(valid_cards) and 
            max(valid_cards) not in partners_cards
        ):
            self.RED_FLAG_TOTAL_COUNT[1] += 1
            self.ALL_RED_FLAG_TOTAL_COUNT[1] += 1
            if card < max(valid_cards):
                self.RED_FLAG_VIOLATIONS[1] += 1
                self.ALL_RED_FLAG_VIOLATIONS[1] += 1

    """
    This function checks if the agent plays a higher card even though their partner is guaranteed to win.
    """
    def partner_win(self, valid_cards, agent_valid_cards, card, cards_played, partners_cards):
        if (len(cards_played) == 3 and len(agent_valid_cards) > 1 and 
            max(valid_cards) in partners_cards
        ):
            self.RED_FLAG_TOTAL_COUNT[2] += 1
            self.ALL_RED_FLAG_TOTAL_COUNT[2] += 1
            if card > max(valid_cards):
                self.RED_FLAG_VIOLATIONS[2] += 1
                self.ALL_RED_FLAG_VIOLATIONS[2] += 1

    """
    This function checks for any red flags based on what the agent played.
    """
    def assess_card_played(self, hands, card, suit, cards_played, player_idx, partners_cards):
        all_valid_cards = list(range(suit * 13, (suit + 1) * 13))
        valid_cards = np.array([i for i in all_valid_cards if i in cards_played])
        agent_valid_cards = np.array([i for i in all_valid_cards if i in hands[player_idx]])

        if suit == -1:
            return

        # highest card played so far is higher than agent's highest card
        self.highest_card(valid_cards, agent_valid_cards, card)
        # 3 cards played and agent has higher cards, does it play highest card or highest necessary card?
        self.higher_card(valid_cards, agent_valid_cards, card, cards_played, partners_cards)
        # 3 cards played + partner has played highest card, does agent play lowest card? do they beat their partner?
        self.partner_win(valid_cards, agent_valid_cards, card, cards_played, partners_cards)


"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"
" PLAY A SINGLE GAME OF BRIDGE
"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
"""
This function plays 13 rounds of 1 NT bridge and outputs a winner.
"""
def play_game(game, hands, train=False, ba=None, barf=None):
    partners_cards = copy.copy(hands[0])
    agents_cards = copy.copy(hands[2])

    declarer, d = agent_declarer(hands)

    """
    hands[0] = North's cards
    hands[1] = East's cards
    hands[2] = Agent's cards
    hands[3] = West's cards
    """

    round_winner = (d + 1) % 4 # the person to the right of the declarer starts the game
    NS_Wins = 0 # used to count total wins in agent partnership

    states_accessed = [] # records which states have been updated for this game
    # For each round
    for _ in range(13):
        cards_played = []
        agent_card_played = [-1, -1]
        agent_state = None
        agent_state_2 = None
        opening_suit = -1

        # Each player plays a card in order starting from round_winner
        for player in range(4):
            card = None
            player_idx = (round_winner + player) % 4
            if player_idx == 2:  # Agent plays
                if ba:
                    agent_state = State(opening_suit, frozenset(cards_played), agent_card_played[1])
                    states_accessed.append(agent_state)
                    card = ba.play_card(agent_state, hands[player_idx])
                else:
                    card = play_random_card(opening_suit, hands[player_idx])
                agent_card_played[0] = card
                barf.assess_card_played(hands, card, opening_suit, cards_played, player_idx, partners_cards)
            elif player_idx == 0: # if agent is declarer, they play their partner's cards
                if ba and declarer:
                    agent_state_2 = State(opening_suit, frozenset(cards_played), agent_card_played[0])
                    states_accessed.append(agent_state_2)
                    card = ba.play_card(agent_state_2, hands[player_idx])
                    barf.assess_card_played(hands, card, opening_suit, cards_played, player_idx, partners_cards)
                else:
                    card = play_random_card(opening_suit, hands[player_idx])
                agent_card_played[1] = card
            else:  # Random bot plays
                card = play_random_card(opening_suit, hands[player_idx])
            # Keep track of the opening suit.
            if player == 0:
                opening_suit = card // 13

            hands[player_idx].remove(card)
            cards_played.append(card)

        # Get the winning card.
        round_winner = (determine_round_winner(opening_suit, cards_played) + round_winner) % 4

        # Adjust the BridgeAgent weights.
        # If the BridgeAgent or N wins.
        if round_winner == 0 or round_winner == 2:
            if ba and train:
                ba.add_win(agent_state, agent_card_played[0])
                if declarer:
                    ba.add_win(agent_state_2, agent_card_played[1])
            NS_Wins += 1
        else:
            if ba and train:
                ba.add_loss(agent_state, agent_card_played[0])
                if declarer:
                    ba.add_loss(agent_state_2, agent_card_played[1])

    # for the last game, determine and write out policy
    if ba and game == (NUM_GAMES_TRAIN - 1):
        policy = []
        count = 0
        for x in ba.weights:
            y = copy.deepcopy(ba.weights[x])
            max = np.argmax(y)
            while max in x.cards_played:
                y[max] = -1
                max = np.argmax(y)
            policy.append(max)
            count += 1
        game_file = "Bridge_" + str(game + 1)
        ba.write_policy(agents_cards, policy, game_file, states_accessed)
    
    return NS_Wins

def game_summary(ba, t, iterations=NUM_GAMES_TRAIN):
    with open(str(NUM_GAMES_TRAIN) + "_Game_Data_Train-" + str(t) + ".csv", 'w') as k:    
        k.write("game,"
                "agent_wins,random_wins,diff_wins,"
                "agent_rfv_a,agent_rftc_a,"
                "agent_rfv_b,agent_rftc_b,"
                "agent_rfv_c,agent_rftc_c,"
                "random_rfv_a,random_rftc_a,"
                "random_rfv_b,random_rftc_b,"
                "random_rfv_c,random_rftc_c\n")

    barf = BridgeAgentRedFlags()
    barf_random = BridgeAgentRedFlags()
    NS_Wins = [0]
    NS_Wins_random = [0]
    
    for game in range(iterations):

        hands = shuffle_cards()

        NS_Wins[-1] += play_game(game=game, hands=copy.deepcopy(hands), train=True, ba=ba, barf=barf)
        NS_Wins_random[-1] += play_game(game=game, hands=hands, ba=None, barf=barf_random)
        ba.game_num += 1

        if (game + 1) % STATS_PER == 0:
            print(f"{game + 1} / ", end="", flush=True)
            rfv = barf.RED_FLAG_VIOLATIONS
            rfv_random = barf_random.RED_FLAG_VIOLATIONS
            rftc = barf.RED_FLAG_TOTAL_COUNT
            rftc_random = barf_random.RED_FLAG_TOTAL_COUNT

            with open(str(NUM_GAMES_TRAIN) + "_Game_Data_Train-" + str(t) + ".csv", 'a') as k:    
                k.write(
                    f"{game + 1},"
                    f"{NS_Wins[-1]},{NS_Wins_random[-1]},{NS_Wins[-1] - NS_Wins_random[-1]},"
                    f"{rfv[0]},{rftc[0]},"
                    f"{rfv[1]},{rftc[1]},"
                    f"{rfv[2]},{rftc[2]},"
                    f"{rfv_random[0]},{rftc_random[0]},"
                    f"{rfv_random[1]},{rftc_random[1]},"
                    f"{rfv_random[2]},{rftc_random[2]},"
                    f"\n")

            # Cumulative statistics on red flags for every STATS_PER games.
            barf.clear_red_flags()
            barf_random.clear_red_flags()
            NS_Wins.append(0)
            NS_Wins_random.append(0)

    average_win_delta = (sum(NS_Wins)-sum(NS_Wins_random)) / ((len(NS_Wins) - 1) * STATS_PER)
    average_rf_ratios_agent = np.divide(barf.ALL_RED_FLAG_VIOLATIONS, barf.ALL_RED_FLAG_TOTAL_COUNT)
    average_rf_ratios_random = np.divide(barf_random.ALL_RED_FLAG_VIOLATIONS, barf_random.ALL_RED_FLAG_TOTAL_COUNT)
    print(f"Average Win Delta (want this to be positive): {average_win_delta}")
    print(f"Average Red Flag Ratios - Agent: {average_rf_ratios_agent}")
    print(f"Average Red Flag Ratios - Random: {average_rf_ratios_random}")
    
    with open(str(NUM_GAMES_TRAIN) + "_Game_Data_Avg_Train-" + str(t) + ".csv", 'w') as m:
        m.write(f"avg_win_delta,avg_rf_agent,avg_rf_random\n"
                f"{average_win_delta},{average_rf_ratios_agent},{average_rf_ratios_random}\n")

    return ba

def main():
    start_time = datetime.datetime.now()
    hands = []

    # TRAINING
    print(f"TRAINING on {NUM_GAMES_TRAIN} games")

    ba = BridgeAgent()
    ba = game_summary(ba, True)

    # TESTING -- we don't change the weights here
    print(f"TESTING on {NUM_GAMES_TEST} games")
    game_summary(ba, False, iterations=NUM_GAMES_TEST)


    end_time = datetime.datetime.now()
    print("Runtime: ", end_time - start_time)  # runtime


if __name__ == "__main__":
    main()
