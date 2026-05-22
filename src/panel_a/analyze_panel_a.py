from __future__ import annotations
import pandas as pd
from scipy.stats import spearmanr

def analyze_panel_a(metrics_csv: str) -> dict:
    d=pd.read_csv(metrics_csv)
    d['nonv']=d['R_cert']<1
    q_m=d['m_hat'].quantile([1/3,2/3]); q_f=d['F_hat'].quantile([1/3,2/3])
    def bucket(r):
        hm=r['m_hat']>=q_m.iloc[1]; lf=r['F_hat']<=q_f.iloc[0]; ln=r['N']>=32
        lm=r['m_hat']<=q_m.iloc[0]; hf=r['F_hat']>=q_f.iloc[1]; sn=r['N']<32
        if hm and lf and ln: return 'high-m/low-F/large-N'
        if lm and hf and sn: return 'low-m/high-F/small-N'
        return 'other'
    d['stratum']=d.apply(bucket,axis=1)
    strat=d.groupby('stratum')['nonv'].mean().to_dict()
    base=[]
    for name,col in [('certificate','R_cert'),('hoeffding','R_hoeffding'),('cantelli','R_cantelli')]:
        cov=(d[col]>=d['R_MC']).mean(); slack=(d.loc[d[col]>=d['R_MC'],col]-d.loc[d[col]>=d['R_MC'],'R_MC']).mean(); nonv=(d[col]<1).mean(); sp=float(spearmanr(d[col],d['R_MC']).correlation)
        base.append({'method':name,'undercoverage_rate':1-cov,'mean_slack_cond_coverage':float(slack),'nonvacuity_rate':float(nonv),'spearman_with_R_MC':sp})
    return {'stratified_nonvacuity':strat,'baseline_comparison':base}
