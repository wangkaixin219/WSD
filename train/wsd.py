from collections import defaultdict
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from queue import PriorityQueue
from brain import *
import time
import numpy as np


class WSD(object):
    def __init__(self, stream, size=10000, checkpoint_path=None):
        self.stream = stream
        self.reservoir = PriorityQueue(maxsize=size)
        self.edges = defaultdict(dict)
        self.vertices = defaultdict(dict)

        self.t = 1
        self.tau = 0
        self.counter = 0
        self.error = 0
        self.mare = 0
        self.c = set()

        self.ground_truth = []

        self.brain = Brain(state_dim=6, action_dim=1)

        if checkpoint_path:
            self.brain.load_model(checkpoint_path=checkpoint_path)
            # print(self.brain.policy.state_dict().values())
            self.kernel1, self.bias1, self.kernel2, self.bias2 = self.brain.policy.actor.state_dict().values()
            self.kernel1 = self.kernel1.numpy()
            self.kernel2 = self.kernel2.numpy()
            self.bias1 = self.bias1.numpy()
            self.bias2 = self.bias2.numpy()
            # print(self.kernel1.shape, self.kernel2.shape)

            self.triangle()

    def triangle(self):
        self.ground_truth.clear()
        neighbor = defaultdict(set)
        counter = 0

        for edge in self.stream:
            v1, v2 = edge
            c = neighbor[v1] & neighbor[v2]
            counter += len(c)
            neighbor[v1].add(v2)
            neighbor[v2].add(v1)
            self.ground_truth.append(counter)

    def reset(self):
        self.triangle()
        self.c = set()
        self.t = 1
        self.tau = 0
        self.counter = 0
        self.error = 0
        self.mare = 0
        self.reservoir.queue.clear()
        self.edges.clear()
        self.vertices.clear()

        state = [1, 1, 0, 1, 1, 1]
        return state

    def weight(self, feat):
        feat = np.array(feat)
        l1 = np.maximum(0, np.dot(self.kernel1, feat) + self.bias1)
        edge_weight = np.maximum(0, np.dot(self.kernel2, l1) + self.bias2) + 1
        return edge_weight.squeeze()

    def get_edge_attr(self, edge, attr):
        v1, v2 = edge
        value = self.edges[edge][attr] if edge in self.edges.keys() else self.edges[(v2, v1)][attr]
        return value

    def set_edge_attr(self, edge, attr, value):
        v1, v2 = edge
        if edge in self.edges.keys():
            self.edges[edge][attr] = value
        else:
            self.edges[(v2, v1)][attr] = value

    def add_edge_attr(self, edge, attr, value):
        v1, v2 = edge
        if edge in self.edges.keys():
            self.edges[edge][attr] += value
        else:
            self.edges[(v2, v1)][attr] += value

    def node_ready(self, list_v):
        for v in list_v:
            if v not in self.vertices:
                self.vertices[v] = {
                    'neighbor_in_sample': set(),
                    'nearest_appear_at': 0,
                    'n_neighbor_in_graph': 0
                }

    def edge_ready(self, edge, u, w, info):
        v1, v2 = edge
        self.edges[edge] = {
            'random': u,
            'weight': w,
            'appear_at': self.t,
            'covariance': 0,
            'estimation': len(self.c),
            'info': info
        }
        self.vertices[v1]['neighbor_in_sample'].add(v2)
        self.vertices[v2]['neighbor_in_sample'].add(v1)
        self.vertices[v1]['nearest_appear_at'] = self.t
        self.vertices[v2]['nearest_appear_at'] = self.t

    def parse_info(self, info):
        _, action, _ = info
        if self.t == 1:
            action = torch.FloatTensor([[1.0]])
        return action.detach().cpu().numpy().flatten().squeeze()

    def step(self, info):
        # take action
        v1, v2 = edge = self.stream[self.t - 1]
        self.node_ready([v1, v2])
        self.vertices[v1]['n_neighbor_in_graph'] += 1
        self.vertices[v2]['n_neighbor_in_graph'] += 1
        u = random.random()
        w = self.parse_info(info)
        r = w / u

        if not self.reservoir.full():
            self.reservoir.put((r, edge))
            self.edge_ready(edge, u, w, info)
        else:
            if r > self.tau:
                r_min, edge_min = self.reservoir.get()
                u1, u2 = edge_min
                self.vertices[u1]['neighbor_in_sample'].remove(u2)
                self.vertices[u2]['neighbor_in_sample'].remove(u1)
                self.edges.pop(edge_min)

                # add new edge
                self.reservoir.put((r, edge))
                self.edge_ready(edge, u, w, info)
                self.tau, _ = self.reservoir.queue[0]

        # next state
        self.t += 1
        v1, v2 = self.stream[self.t - 1]
        self.node_ready([v1, v2])
        self.c = self.vertices[v1]['neighbor_in_sample'] & self.vertices[v2]['neighbor_in_sample']

        # update counter, get reward
        tri_t = 0
        for v in self.c:
            w1 = self.get_edge_attr(edge=(v, v1), attr='weight')
            w2 = self.get_edge_attr(edge=(v, v2), attr='weight')
            t1 = self.get_edge_attr(edge=(v, v1), attr='appear_at')
            t2 = self.get_edge_attr(edge=(v, v2), attr='appear_at')

            p1 = 1 if self.tau == 0 else min(1, w1 / self.tau)
            p2 = 1 if self.tau == 0 else min(1, w2 / self.tau)

            delta = 1 / (p1 * p2)
            self.counter += delta

            self.add_edge_attr(edge=(v, v1), attr='estimation', value=delta)
            self.add_edge_attr(edge=(v, v2), attr='estimation', value=delta)

            tri_t = max(tri_t, t1, t2)

        error = np.abs(self.counter - self.ground_truth[self.t - 1])
        reward = self.error - error
        state, action, action_logprob = info
        self.brain.store_sample(state, action, action_logprob, reward, False)
        self.error = error
        self.mare += self.error / (self.ground_truth[self.t - 1] + 1)

        done = False if self.t < len(self.stream) else True

        observation = [
            self.vertices[v1]['n_neighbor_in_graph'],
            self.vertices[v2]['n_neighbor_in_graph'],
            len(self.c),
            self.t + 1,
            max(self.vertices[v1]['nearest_appear_at'], self.vertices[v2]['nearest_appear_at']) + 1,
            tri_t + 1
        ]

        return observation, reward, done

    def train_brain(self):
        epoch = 1
        while epoch < 1000:
            state = self.reset()
            done = False
            while not done:
                info = self.brain.select_action(state)
                state, reward, done = self.step(info)

                if self.t % 20000 == 0:
                    self.brain.update()
            self.brain.update()

            self.brain.save_model(checkpoint_path='model/WSD_{}.pth'.format(epoch))
            print(epoch, self.counter, self.error / (self.ground_truth[self.t - 1] + 1), self.mare / self.t)

            epoch += 1
