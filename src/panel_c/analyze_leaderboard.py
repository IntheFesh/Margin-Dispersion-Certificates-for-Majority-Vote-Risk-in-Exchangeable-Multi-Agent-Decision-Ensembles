from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from src.panel_c.load_leaderboard_predictions import load_leaderboard_predictions
from src.panel_b.pairwise_stats import normalized_correlation

def run(input_dir: str, output_dir: str):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    d=load_leaderboard_predictions(input_dir)
    piv=d.pivot_table(index='instance_id',columns='model_id',values='correct',aggfunc='first')
    fam=d.drop_duplicates('model_id').set_index('model_id')['family']
    models=list(piv.columns)
    rows=[]
    for i,a in enumerate(models):
        for b in models[i+1:]:
            x=piv[a].dropna(); y=piv[b].dropna(); common=x.index.intersection(y.index)
            rho=normalized_correlation(x.loc[common].to_numpy(),y.loc[common].to_numpy())
            rows.append({'model_i':a,'model_j':b,'rho_ij':rho,'family_i':fam[a],'family_j':fam[b],'same_family':fam[a]==fam[b]})
    p=pd.DataFrame(rows); p.to_csv(out/'panel_c_pairwise_correctness.csv',index=False)
    s=p.groupby('same_family')['rho_ij'].mean().rename({True:'within_family_mean_rho',False:'cross_family_mean_rho'}).to_dict()
    gap=float(s.get('within_family_mean_rho',np.nan)-s.get('cross_family_mean_rho',np.nan))
    pd.DataFrame([{'within_family_mean_rho':s.get('within_family_mean_rho',np.nan),'cross_family_mean_rho':s.get('cross_family_mean_rho',np.nan),'family_gap':gap}]).to_csv(out/'panel_c_family_summary.csv',index=False)
    (out/'panel_c_report.md').write_text('# Panel C Supplementary\n\nAppendix-only correctness-correlation analysis; no CKA and no representation claims.\n')
