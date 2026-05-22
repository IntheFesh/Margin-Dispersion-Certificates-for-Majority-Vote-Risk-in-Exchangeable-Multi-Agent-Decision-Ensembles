from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.config.load_config import resolve_runtime_config
from src.panel_c.load_leaderboard_predictions import load_leaderboard_predictions
from src.panel_b.pairwise_stats import normalized_correlation

def run(input_dir: str, output_dir: str):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    d=load_leaderboard_predictions(input_dir)
    piv=d.pivot_table(index='instance_id',columns='model_id',values='correct',aggfunc='first')
    fam=d.drop_duplicates('model_id').set_index('model_id')['family']
    rows=[]; models=list(piv.columns)
    for i,a in enumerate(models):
        for b in models[i+1:]:
            c=piv[[a,b]].dropna(); rows.append({'model_i':a,'model_j':b,'rho_ij':normalized_correlation(c[a].to_numpy(),c[b].to_numpy()),'same_family':fam[a]==fam[b]})
    p=pd.DataFrame(rows); p.to_csv(out/'panel_c_pairwise_correctness.csv',index=False)
    s=p.groupby('same_family')['rho_ij'].mean(); pd.DataFrame([{'within_family_mean_rho':float(s.get(True,np.nan)),'cross_family_mean_rho':float(s.get(False,np.nan)),'family_gap':float(s.get(True,np.nan)-s.get(False,np.nan))}]).to_csv(out/'panel_c_family_summary.csv',index=False)
    (out/'panel_c_report.md').write_text('Panel C supplementary only; no CKA and no representation-alignment claims.\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output_dir'); ap.add_argument('--seed',type=int); ap.add_argument('--validate_only',action='store_true')
    a=ap.parse_args(); cfg=resolve_runtime_config(a.config,a.output_dir,a.seed); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    if a.validate_only or cfg.get('validate_only',False):
        (out/'panel_c_report.md').write_text('Panel C supplementary validate_only; not run due to missing per-instance real predictions.\n'); return
    input_dir=cfg.get('leaderboard_input_dir')
    if not input_dir: raise ValueError('leaderboard_input_dir required')
    run(input_dir,str(out))
if __name__=='__main__': main()
