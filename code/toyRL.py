"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time
import sys
np.random.seed(2)  # reproducible

SIZE = 2
N_STATES = 8   # the length of the 1 dimensional world
ACTIONS = ['nothing','upleft', 'upright','downleft','downright']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 50   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

def indices_array(n):
    r = np.arange(n)
    out = np.empty((n,n,2),dtype=int)
    out[:,:,0] = r[:,None]
    out[:,:,1] = r
    return out

class Game:
    def __init__(self, size, **kwargs):

        self.size = size
        self.counter = 0
        maparg = kwargs.get('map', None)
        if maparg is None:
            self.map = np.random.randint(0,2,(size,size))
        else:
            if maparg.size == size**2:
                self.map = maparg
            else:
                print("Double Check the size or map")

    def printGame(self):
        print(self.map)

    def printemoji(self):
        if self.counter > 0:
            for i in range(self.size):
                sys.stdout.write("\033[F")
        for row in self.map:
            for col in row:
                if col == 0:
                    print("⭕️",end=" ")
                else:
                    print("🐭",end=" ")
            print("")
        self.counter +=1

    def getneibours(self, row, col):
        # max: 4 neibours
        neibours = []
        for i,j in [[row-1,col],[row+1,col],[row,col-1],[row,col+1]]:
            if i>=0 and j>=0 and i<self.size and j<self.size:
                neibours.append([i,j])
        return neibours

    def rm(self,row,col):
        self.map[row][col] = 0

    def add(self,row,col):
        self.map[row][col] += 1
        self.map[row][col] %= 2

    def hit(self, row, col):
        row = row - 1
        col = col - 1
        if row not in range(0,self.size):
            self.printemoji()
            return
        if col not in range(0,self.size):
            self.printemoji()
            return
        if self.map[row][col] != 1:
            # nothing happens when hit the air
            self.printemoji()
            return
        else:
            # hit row,col
            self.rm(row,col)
            [self.add(i,j) for i,j in self.getneibours(row,col)]
            self.printemoji()
            return

    def play(self):
        self.printemoji()
        while sum(sum(self.map))>0 or inputs=='exit':
            inputs = input("Choose row and col (eg. 1 2) or exit:")
            if inputs == 'exit':
                break
            row, col = inputs.split(" ")
            self.hit(int(row),int(col))
        print("Done👍")


def build_q_table(n_states, actions):
    table = pd.DataFrame(columns=actions, dtype=np.float64)
    return table

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.loc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

def get_env_feedback(S, A, game):
    # This is how agent will interact with the environment
    prev = np.array2string(game.map)
    R = -0.1
    if A == 'upleft':    # move right
        game.hit(1,1)
        S_ = np.array2string(game.map)
        gameover = sum(sum(game.map))
        comparison = prev == S_
        if gameover == 0:   # terminate
            S_ = 'terminal'
            R = 2
        elif comparison:
            R = -0.1

    elif A == 'upright':   # move left
        game.hit(1,2)
        S_ = np.array2string(game.map)
        comparison = prev == S_
        gameover = sum(sum(game.map))
        if gameover == 0:   # terminate
            S_ = 'terminal'
            R = 2
        elif comparison:
            R = -0.1

    elif A == 'downleft':    # move right   
        game.hit(2,1)
        S_ = np.array2string(game.map)
        gameover = sum(sum(game.map))
        comparison = prev == S_
        if gameover == 0:   # terminate
            S_ = 'terminal'
            R = 2
        elif comparison:
            R = -0.1
    elif A == 'downright':   # move left
        game.hit(2,2)
        S_ = np.array2string(game.map)
        comparison = prev == S_
        gameover = sum(sum(game.map))
        if gameover == 0:   # terminate
            S_ = 'terminal'
            R = 2
        elif comparison:
            R = -0.1
    elif A == 'nothing':
        S_ = np.array2string(game.map)
        if sum(sum(game.map)) == 0:
            S_ = 'terminal'
            R = -1
        else:
            R = -1

    return S_, R

def check_state_exist(state,q_table,actions):
    if state not in q_table.index:
        # append new state to q table
        q_table = q_table.append(
            pd.Series(
                [0]*len(actions),
                index=q_table.columns,
                name=state,
            )
        )
    return q_table
def update_env(S, episode, step_counter, game):
    # This is how environment be updated

    # env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        # env_list[S] = 'o'
        # interaction = ''.join(env_list)
        game.printemoji()
        # print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        game = Game(size=2)
        S = np.array2string(game.map)
        q_table = check_state_exist(S, q_table,ACTIONS)
        is_terminated = False
        update_env(S, episode, step_counter,game)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A, game)  # take action & get next state and reward
            q_table = check_state_exist(S_, q_table,ACTIONS)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.loc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1, game)
            step_counter += 1
        print("================")
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\nQ-table:\n')
    print(q_table)
