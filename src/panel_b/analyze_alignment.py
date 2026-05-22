from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from src.config.load_config import resolve_runtime_config

def run(pairwise_csv: str, cka_csv: str, null_csv: str, out_dir: str):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(pairwise_csv); c=pd.read_csv(cka_csv)
    m=p.merge(c,on=['model_i','model_j','benchmark'],how='inner')
    agg=float(spearmanr(m['rho_ij'],m['cka']).correlation)
    n=pd.read_csv(null_csv); q99=float(n['stat'].quantile(0.99))
    (out/'panel_b_alignment_summary.json').write_text(json.dumps({'aggregate_spearman':agg,'family_aware_null_q99':q99,'confirmatory_pass':agg>q99,'note':'external grounding only; observational non-causal'},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output_dir'); ap.add_argument('--seed',type=int); ap.add_argument('--validate_only',action='store_true')
    a=ap.parse_args(); cfg=resolve_runtime_config(a.config,a.output_dir,a.seed)
    if a.validate_only or cfg.get('validate_only',False): return
    raise FileNotFoundError('Need pairwise/cka/null CSV files generated from real resources.')
if __name__=='__main__': main()
