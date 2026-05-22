from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config.load_config import resolve_runtime_config
from src.theory.estimators import estimate_basic_summaries
from src.theory.certificate import empirical_certificate_from_X, margin_only_hoeffding_baseline, asymptotic_cantelli_bound
from src.theory.majority_risk import monte_carlo_reference_risk

def _matrix(df,K):
    p=df.pivot_table(index='instance_id',columns='sample_id',values='correct',aggfunc='first').sort_index(axis=1)
    if p.shape[1] < K: raise ValueError('insufficient K_ref columns')
    return p.iloc[:,:K].to_numpy(int)

def run(cfg):
    out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(cfg['full_correctness_csv'])
    rows=[]; boot=[]
    for (bench,proto,reg),g in df.groupby(['benchmark','protocol','regime']):
        X=_matrix(g,int(cfg['K_ref']))
        for K_est in cfg['K_est_values']:
            X_est,X_ref=X[:,:K_est],X[:,K_est:]
            a_ref=X_ref.mean(axis=1); base=estimate_basic_summaries(X_est)
            for N in cfg['N_values']:
                cert=empirical_certificate_from_X(X_est,N,float(cfg['global_delta']))
                R_MC=monte_carlo_reference_risk(N,a_ref)
                margin=base['bar_alpha_hat']
                hoe=margin_only_hoeffding_baseline(margin,N)
                can=asymptotic_cantelli_bound(margin,base['F_hat_clipped'])
                rows.append({'benchmark':bench,'protocol':proto,'regime':reg,'K_est':K_est,'N':N,'R_cert':cert['R_cert'],'R_MC':R_MC,'R_hoeffding':hoe,'R_cantelli':can,'coverage_cert':int(cert['R_cert']>=R_MC),'coverage_hoeffding':int(hoe>=R_MC),'coverage_cantelli':int(can>=R_MC),'m_hat':base['margin_hat'],'F_hat':base['F_hat_unbiased'],'m_L':cert.get('m_L'),'U_F':cert.get('U_F'),'eta_star':cert.get('eta_star')})
                rng=np.random.default_rng(int(cfg['seed'])+N+K_est)
                M=X_est.shape[0]
                for b in range(int(cfg['B_boot'])):
                    idx=rng.integers(0,M,size=M)
                    xb=X_est[idx,:]
                    cb=empirical_certificate_from_X(xb,N,float(cfg['global_delta']))
                    boot.append({'benchmark':bench,'protocol':proto,'regime':reg,'K_est':K_est,'N':N,'boot_id':b,'R_cert_boot':cb['R_cert']})
    m=pd.DataFrame(rows); mb=pd.DataFrame(boot)
    m.to_csv(out/'panel_a_cell_metrics.csv',index=False); mb.to_csv(out/'panel_a_bootstrap_metrics.csv',index=False)
    m[['benchmark','protocol','regime','K_est','N','R_MC','R_cert','R_hoeffding','R_cantelli']].to_csv(out/'panel_a_baseline_comparison.csv',index=False)
    nv=m.groupby(['benchmark','protocol','regime','K_est','N']).agg(nonvacuous=('R_cert',lambda s:float((s<1).mean())),lt07=('R_cert',lambda s:float((s<0.7).mean())),lt03=('R_cert',lambda s:float((s<0.3).mean()))).reset_index()
    nv.to_csv(out/'panel_a_nonvacuity_summary.csv',index=False)
    (out/'panel_a_report.md').write_text('# Panel A Full\n\nEmpirical coverage check (not formal proof).\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output_dir'); ap.add_argument('--seed',type=int); ap.add_argument('--validate_only',action='store_true')
    a=ap.parse_args(); cfg=resolve_runtime_config(a.config,a.output_dir,a.seed)
    if a.validate_only or cfg.get('validate_only',False): return
    run(cfg)
if __name__=='__main__': main()
