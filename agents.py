import random
from collections import namedtuple

import math

import torch
import torch.nn.functional as fn

from torch import nn, optim, LongTensor, FloatTensor, ByteTensor
from torch.autograd import Variable

import numpy as np

from common import Point

Transition = namedtuple('Transition', (
    'state', 'action', 'next_state', 'reward'
))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(128, 4)

    def forward(self, x):
        x = fn.relu(self.bn1(self.conv1(x)))
        x = fn.relu(self.bn2(self.conv2(x)))
        x = fn.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


def calc_reward(snake, world):
    if not snake.alive:
        return 0, True
    elif np.sum(world % 10 > 0) > len(snake.pieces):
        return 0, False
    else:
        return (len(snake.pieces) - 2), True


MOVES = [
    Point(-1, 0),
    Point(1, 0),
    Point(0, -1),
    Point(0, 1),
]


def world_to_state(world: np.ndarray):
    s = torch.from_numpy(world.copy())
    return s.unsqueeze(0).type(FloatTensor).unsqueeze(0)


class RLAgent:
    def __init__(self, init_state: np.ndarray):
        self.model = DQNetwork()
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(10_000)
        self.steps_done = 0
        self.current_state = world_to_state(init_state)
        self.next_state = None
        self.last_action = None
        self.episode_reward = 0

    def reset(self, world: np.ndarray):
        self.current_state = world_to_state(world)
        self.next_state = None
        self.last_action = None
        print(self.episode_reward)
        self.episode_reward = 0

    def get_move(self, snake, world):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            state = world_to_state(world)
            self.last_action = self.model(
                Variable(state, volatile=True).type(FloatTensor)
            ).data.max(1)[1].view(1, 1)
        else:
            self.last_action = LongTensor([[random.randrange(4)]])
        return MOVES[self.last_action[0][0]]

    def update(self, snake, world: np.ndarray):
        state = world_to_state(world)
        reward, done = calc_reward(snake, world)
        self.episode_reward += reward
        reward = FloatTensor([reward])
        if not done:
            self.next_state = state
        else:
            self.next_state = None
        self.memory.push(self.current_state, self.last_action, self.next_state, reward)
        self.current_state = self.next_state
        self.optimize_model()
        return not done

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = fn.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
