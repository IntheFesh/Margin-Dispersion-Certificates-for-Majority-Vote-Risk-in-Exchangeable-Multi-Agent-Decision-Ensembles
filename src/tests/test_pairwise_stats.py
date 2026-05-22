import numpy as np
from src.panel_b.pairwise_stats import pairwise_q_statistic

def test_q_range():
 q=pairwise_q_statistic(np.array([1,1,0,0]),np.array([1,0,1,0]))
 assert -1<=q<=1
