from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


def run(pairwise_csv: str, cka_csv: str, null_csv: str, out_dir: str):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(pairwise_csv); c=pd.read_csv(cka_csv)
    m=p.merge(c,on=['model_i','model_j','benchmark'],how='inner')
    agg=float(spearmanr(m['rho_ij'],m['cka']).correlation)
    by={b:float(spearmanr(g['rho_ij'],g['cka']).correlation) for b,g in m.groupby('benchmark')}
    n=pd.read_csv(null_csv)
    q99=float(n['stat'].quantile(0.99))
    summary={'aggregate_spearman':agg,'benchmark_spearman':by,'family_aware_null_q99':q99,'confirmatory_pass':agg>q99,'panel_b_note':'External grounding only; observational and non-causal.'}
    (out/'panel_b_alignment_summary.json').write_text(json.dumps(summary,indent=2))
