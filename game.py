import pygame
import sys
from pygame.locals import *
from collections import namedtuple, deque
import numpy as np
import random
from agents import RLAgent
from common import Point

WINDOW_SIZE = 400
TILE_SIZE = 20
TILE_WIDTH = WINDOW_SIZE // TILE_SIZE
pygame.init()
pygame.mixer.quit()
s = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Battlesnake?')
clock = pygame.time.Clock()


class Snake:
    def __init__(self, length=3, start_pos: Point = None, world: np.array = None):
        if start_pos is None:
            start_pos = Point(10, 10)
        self.pieces = deque([start_pos] * length)
        self.world = world
        self.health = 100
        world[start_pos.x, start_pos.y] += length
        self.alive = True

    def move(self, move):
        tail = self.pieces[-1]
        self.world[tail.x, tail.y] -= 1

        self.pieces.rotate(1)
        self.pieces[0] = self.pieces[1] + move

        head = self.pieces[0]
        w, h = self.world.shape
        if 0 <= head.x < w and 0 <= head.y < h:
            self.world[head.x, head.y] += 1

    def head(self):
        return self.pieces[0]


class DumbAgent:
    def __init__(self):
        pass

    def reset(self, world):
        pass

    def update(self, snake, world):
        return True

    def get_move(self, snake, world):
        moves = []
        head = snake.head()
        w, h = world.shape
        if head.x > 1:
            moves.append(Point(-1, 0))
        if head.x < w - 1:
            moves.append(Point(1, 0))
        if head.y > 1:
            moves.append(Point(0, -1))
        if head.y < h - 1:
            moves.append(Point(0, 1))
        next_seg = snake.pieces[1] - head
        if next_seg in moves:
            moves.remove(next_seg)
        return random.choice(moves)


class Game:
    def __init__(self):
        self.world = np.zeros((TILE_SIZE, TILE_SIZE))
        self.snakes = [
            # Snake(start_pos=self.find_empty_space(), world=self.world),
            # Snake(start_pos=self.find_empty_space(), world=self.world),
            Snake(start_pos=self.find_empty_space(), world=self.world),
            Snake(start_pos=self.find_empty_space(), world=self.world)
        ]
        self.agents = [
            # DumbAgent(),
            # DumbAgent(),
            DumbAgent(),
            RLAgent(self.world),
        ]
        self.food = []
        for i in range(3):
            self.add_food()

    def reset(self):
        self.world = np.zeros((TILE_SIZE, TILE_SIZE))
        self.snakes = [
            # Snake(start_pos=self.find_empty_space(), world=self.world),
            # Snake(start_pos=self.find_empty_space(), world=self.world),
            Snake(start_pos=self.find_empty_space(), world=self.world),
            Snake(start_pos=self.find_empty_space(), world=self.world)
        ]
        self.food = []
        for i in range(3):
            self.add_food()

    def find_empty_space(self):
        open_spaces = np.argwhere(self.world == 0)
        return Point(*random.choice(open_spaces))

    def add_food(self):
        f = self.find_empty_space()
        self.world[f.x, f.y] += 10
        self.food.append(f)

    def reset_food(self, f):
        self.food.remove(f)
        self.world[f.x, f.y] -= 10
        self.add_food()

    def step(self):
        for (agent, snake) in zip(self.agents, self.snakes):
            if snake.alive:
                snake.move(agent.get_move(snake, self.world))

        self.collide()
        self.render()

        for (agent, snake) in zip(self.agents, self.snakes):
            if not agent.update(snake, self.world):
                self.reset()
                for a in self.agents:
                    a.reset(self.world)
                break

    def render(self):
        for (i, snake) in enumerate(self.snakes):
            if not snake.alive:
                continue
            col = 0xf * (i + 1) * 4
            img.fill((0, 0, col))
            for p in snake.pieces:
                s.blit(img, (p.x * TILE_WIDTH + 1, p.y * TILE_WIDTH + 1))
            head.fill((col, 0, col))
            s.blit(head, (snake.head().x * TILE_WIDTH + 1, snake.head().y * TILE_WIDTH + 1))
        for f in self.food:
            s.blit(food, (f.x * TILE_WIDTH + 4, f.y * TILE_WIDTH + 4))

    def collide(self):
        dead = set()
        for snake in self.snakes:
            if not snake.alive:
                continue

            if snake.health == 0:
                dead.add(snake)
            else:
                snake.health -= 1

            head = snake.head()
            w, h = self.world.shape
            if head.x < 0 or head.x >= w or head.y < 0 or head.y >= h:
                dead.add(snake)
            if head in list(snake.pieces)[1:]:
                dead.add(snake)

            for other in self.snakes:
                if snake != other:
                    if head == other.head():
                        slen, othlen = len(snake.pieces), len(other.pieces)
                        if slen >= othlen:
                            dead.add(other)
                        if slen <= othlen:
                            dead.add(snake)
                    elif head in other.pieces:
                        dead.add(snake)

            if snake not in dead:
                for f in self.food:
                    if head == f:
                        tail = snake.pieces[-1]
                        snake.pieces.append(tail)
                        snake.health = 100
                        self.world[tail.x, tail.y] += 1
                        self.reset_food(f)

        for s in dead:
            s.alive = False
            for p in s.pieces:
                if 0 <= p.x < TILE_SIZE and 0 <= p.y < TILE_SIZE:
                    self.world[p.x, p.y] -= 1


BG_COL = (0xff, 0xff, 0xff)

elapsed = 0

frame_time = 250

if __name__ == '__main__':
    img = pygame.Surface((TILE_WIDTH - 2, TILE_WIDTH - 2))
    img.fill((0, 0, 0xff))
    head = pygame.Surface((TILE_WIDTH - 2, TILE_WIDTH - 2))
    head.fill((0xff, 0, 0xff))
    food = pygame.Surface((TILE_WIDTH - 8, TILE_WIDTH - 8))
    food.fill((0, 0xff, 0))
    game = Game()
    while True:
        elapsed += clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit(0)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_UP and frame_time > 25:
                    frame_time -= 25
                elif e.key == pygame.K_DOWN and frame_time < 1000:
                    frame_time += 25

        if elapsed >= frame_time:
            s.fill(BG_COL)
            game.step()
            elapsed = 0

        pygame.display.flip()
