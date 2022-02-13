from utils import Graph
from wsd import WSD as WSD


reservoir_size = 200000 

g = Graph('../dataset/sample.edges')

wsd = WSD(g.stream, size=reservoir_size)
wsd.train_brain()

