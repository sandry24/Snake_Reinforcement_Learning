import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(16, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_d = Point(head.x, head.y + 20)
        point_u = Point(head.x, head.y - 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN
        dir_u = game.direction == Direction.UP

        state = [
            # Danger straight
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Coords
            game.head.x / game.w,
            (game.w - game.head.x - 1) / game.w,
            game.head.y / game.h,
            (game.h - game.head.y - 1) / game.h,

            # Angle between head and food
            math.degrees(math.atan2(game.food.y - game.head.y, game.food.x - game.head.x)) / 360
        ]
        # 0 is free cell, 0.33 snake body, 0.66 snake head, 1 food
        # for x in range(0, game.w+1, 20):
        #     for y in range(0, game.h+1, 20):
        #         if Point(x, y) in game.snake[1:]:
        #             state.append(0.33)
        #         elif Point(x, y) == game.food:
        #             state.append(1)
        #         elif Point(x, y) == game.head:
        #             state.append(0.66)
        #         else:
        #             state.append(0)
        # min_left = float(game.head.x)
        # min_right = game.w - game.head.x - 1
        # min_up = float(game.head.y)
        # min_down = game.h - game.head.y - 1
        # for segment in game.snake[1:]:
        #     if segment.y == game.head.y:
        #         if segment.x < game.head.x:
        #             min_left = min(min_left, game.head.x - segment.x)
        #         else:
        #             min_right = min(min_right, segment.x - game.head.x)
        #     elif segment.x == game.head.x:
        #         if segment.y < game.head.y:
        #             min_up = min(min_up, game.head.y - segment.y)
        #         else:
        #             min_down = min(min_down, segment.y - game.head.y)
        # state.extend([min_left/game.w, min_right/game.w, min_up/game.h, min_down/game.h])
        # print(state)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration/exploitation
        self.epsilon = 500 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 500) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("model5.pth")

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()