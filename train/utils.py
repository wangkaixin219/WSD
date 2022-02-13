from collections import defaultdict
import random


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Graph(object):
    def __init__(self, filename):
        self.stream = []
        self.counter = 0
        print('Read graph ' + filename)
        with open(filename, 'r') as graph:
            for line in graph.readlines():
                if line.startswith('%'):
                    continue
                v1, v2 = line.strip().split(',')
                v1, v2 = int(v1), int(v2)
                if v1 == v2:
                    continue
                self.stream.append((v1, v2))
        print('Stream data complete')

