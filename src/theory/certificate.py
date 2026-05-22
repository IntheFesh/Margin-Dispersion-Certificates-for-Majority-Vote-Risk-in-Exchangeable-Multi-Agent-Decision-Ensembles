from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize_scalar
from .estimators import estimate_m1_m2

def compute_confidence_bounds(m1,m2,M,delta):
    if not (0<delta<1): raise ValueError('delta must be in (0,1)')
    if M<1: raise ValueError('M must be >=1')
    eps=math.sqrt(math.log(4/delta)/(2*M)); L_alpha=m1-eps; U2=m2+eps
    U_F=min(0.25,max(0.0,U2-L_alpha**2)); m_L=L_alpha-0.5
    return {"L_alpha":L_alpha,"U2":U2,"U_F":U_F,"m_L":m_L,"eps_delta":eps}

def certificate_objective(eta,margin,F_or_UF,N):
    if not (0<eta<margin): raise ValueError('eta must satisfy 0<eta<margin')
    return F_or_UF/(F_or_UF+(margin-eta)**2)+math.exp(-2*N*eta**2)

def optimize_eta(margin,F_or_UF,N):
    if margin<=0: return {"eta_star":None,"objective_value":1.0,"issued":False}
    eps=min(1e-8,margin/1000) if margin>1e-12 else margin/10
    lo,hi=eps,max(margin-eps,eps*2)
    try:
        r=minimize_scalar(lambda e: certificate_objective(e,margin,F_or_UF,N),bounds=(lo,hi),method='bounded')
        if not r.success: raise RuntimeError
        eta,val=float(r.x),float(r.fun)
    except Exception:
        grid=np.linspace(lo,hi,2000); vals=np.array([certificate_objective(g,margin,F_or_UF,N) for g in grid])
        i=int(vals.argmin()); eta,val=float(grid[i]),float(vals[i])
    return {"eta_star":eta,"objective_value":min(1.0,val),"issued":True}

def population_certificate(bar_alpha,F,N):
    m=bar_alpha-0.5
    if m<=0: return {"issued":False,"R_cert":1.0,"eta_star":None,"reason":"m <= 0; no favorable-side certificate"}
    opt=optimize_eta(m,F,N)
    return {"issued":True,"R_cert":opt['objective_value'],"eta_star":opt['eta_star'],"margin":m,"F":F}

def empirical_certificate_from_summaries(m1,m2,M,N,delta):
    b=compute_confidence_bounds(m1,m2,M,delta)
    if b['m_L']<=0: return {**b,"issued":False,"R_cert":1.0,"eta_star":None,"reason":"m_L <= 0; no favorable-side certificate"}
    opt=optimize_eta(b['m_L'],b['U_F'],N)
    return {**b,"issued":True,"R_cert":opt['objective_value'],"eta_star":opt['eta_star'],"reason":"issued"}

def empirical_certificate_from_X(X,N,delta):
    s=estimate_m1_m2(X)
    return empirical_certificate_from_summaries(s['m1'],s['m2'],len(s['Z']),N,delta)

def margin_only_hoeffding_baseline(bar_alpha,N):
    m=bar_alpha-0.5
    return 1.0 if m<=0 else min(1.0,math.exp(-2*N*m*m))

def asymptotic_cantelli_bound(bar_alpha,F):
    m=bar_alpha-0.5
    return 1.0 if m<=0 else min(1.0,F/(F+m*m))
