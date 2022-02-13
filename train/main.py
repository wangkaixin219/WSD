import random
import queue
from utils import Graph
import random
from wsd import WSD as WSD
import numpy as np
import time


reservoir_size = 1e4

g = Graph('../datasets/com-youtube.edges')

wsd = WSD(g.stream, size=reservoir_size)
wsd.train_brain()

