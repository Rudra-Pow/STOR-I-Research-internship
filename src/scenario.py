import numpy as np

class Scenario:
    def __init__(self, prob: float, q, h, T, W):
        self.prob = float(prob)
        self.q = np.asarray(q, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.W = np.asarray(W, dtype=float)
