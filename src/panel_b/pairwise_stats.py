import numpy as np

def pairwise_q_statistic(x,y):
 x=np.asarray(x); y=np.asarray(y)
 if x.shape!=y.shape: raise ValueError("shape mismatch")
 if not np.all(np.isin(x,[0,1])) or not np.all(np.isin(y,[0,1])): raise ValueError("binary required")
 n11=np.sum((x==1)&(y==1)); n00=np.sum((x==0)&(y==0)); n10=np.sum((x==1)&(y==0)); n01=np.sum((x==0)&(y==1))
 d=n11*n00+n10*n01
 return 0.0 if d==0 else float((n11*n00-n10*n01)/d)
