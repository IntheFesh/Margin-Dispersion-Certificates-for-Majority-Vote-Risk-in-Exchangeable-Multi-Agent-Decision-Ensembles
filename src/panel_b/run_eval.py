from __future__ import annotations
import argparse
from pathlib import Path
from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report

def run_eval_from_predictions(predictions_csv: str, output_csv: str) -> None:
    import pandas as pd
    df=pd.read_csv(predictions_csv)
    req=['instance_id','benchmark','model_id','family','scale','raw_output','parsed_answer','gold_answer','correct','invalid_parse']
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f'Missing required columns: {miss}')
    df[req].to_csv(output_csv,index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output_dir'); ap.add_argument('--seed',type=int); ap.add_argument('--validate_only',action='store_true')
    a=ap.parse_args(); cfg=resolve_runtime_config(a.config,a.output_dir,a.seed); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    if a.validate_only or cfg.get('validate_only',False):
        write_markdown_report(str(out/'panel_b_report.md'),'Panel B Report','validate_only',a.config,cfg,int(cfg['seed']),0,0,0.0,'validate_only run',[],['not run due to missing real resource'],[])
        return
    raise FileNotFoundError('Real Panel B prediction inputs are required; no fake inference is allowed.')
if __name__=='__main__': main()
