from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_csv',required=True); ap.add_argument('--output_dir',required=True)
    a=ap.parse_args(); p=Path(a.input_csv)
    if not p.exists(): raise FileNotFoundError(f'Missing input file: {p}')
    df=pd.read_csv(p)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(5,4));
    cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols)<1: raise ValueError('No numeric columns for plotting')
    x=df[cols[0]].to_numpy(); y=df[cols[1]].to_numpy() if len(cols)>1 else df.index.to_numpy()
    ax.scatter(x,y,s=12)
    ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1] if len(cols)>1 else 'index')
    ax.set_title('certified low-risk / intermediate / uninformative regions')
    stem=Path(__file__).stem
    fig.tight_layout(); fig.savefig(out/f'{stem}.png'); fig.savefig(out/f'{stem}.pdf')
if __name__=='__main__': main()
