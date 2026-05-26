#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/Margin-Dispersion

export HF_HOME=/root/autodl-tmp/models
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/models
mkdir -p outputs/panel_a outputs/pilots outputs/logs

# (hf_path, model_key, port, gpu_util, max_seqs, max_batched_tokens, extra_args)
MODELS=(
  #"/root/autodl-tmp/models/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct|8000|0.92|512|8192|--trust-remote-code"
  "/root/autodl-tmp/models/Qwen2.5-3B-Instruct|qwen2.5-3b-instruct|8001|0.92|512|8192|"
  "/root/autodl-tmp/models/Qwen2.5-14B-Instruct|qwen2.5-14b-instruct|8002|0.92|512|8192|--trust-remote-code"
)
BENCHMARKS=("arc_challenge" "gsm8k" "mmlu_subset")
SEED=51966  # 0xCAFE

start_vllm() {
  local hf_path="$1" port="$2" gpu_util="$3" max_seqs="$4" max_batched="$5" extra="$6"
  echo "[vllm] starting $hf_path on port $port (max_seqs=$max_seqs, gpu=$gpu_util)"
  vllm serve "$hf_path" \
    --port "$port" --gpu-memory-utilization "$gpu_util" --max-model-len 4096 \
    --seed 0 --enable-prefix-caching \
    --max-num-seqs "$max_seqs" --max-num-batched-tokens "$max_batched" \
    --dtype bfloat16 \
    --download-dir /root/autodl-tmp/models/_vllm_cache \
    $extra > "outputs/logs/vllm_${port}.log" 2>&1 &
  echo $!
}

wait_vllm() {
  local port="$1"
  for i in $(seq 1 300); do
    if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
      echo "[vllm] port $port ready after ${i}s"
      return 0
    fi
    sleep 2
  done
  echo "[vllm] port $port TIMEOUT after 600s; check outputs/logs/vllm_${port}.log"
  return 1
}

stop_vllm() {
  local pid="$1"
  kill -SIGINT "$pid" 2>/dev/null || true
  for i in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  # 强杀整个进程组（vLLM 有 APIServer + EngineCore 子进程）
  pkill -9 -P "$pid" 2>/dev/null || true
  kill -9 "$pid" 2>/dev/null || true
  # 杀所有 GPU 上的残留进程
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
  # 等显存真正释放（最多 30 秒）
  for i in $(seq 1 30); do
    local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$mem" -lt 1000 ]; then
      echo "[vllm] GPU released after ${i}s (mem=${mem} MiB)"
      return 0
    fi
    sleep 1
  done
  echo "[vllm] WARNING: GPU mem still high after 30s"
}

# 并发版 A1 执行器（关键：用 asyncio 让 vLLM batching 起作用）
run_a1_cell_concurrent() {
  local bench="$1" model_key="$2" hf_path="$3" port="$4" pool_type="$5" K_arg="$6" cell_dir="$7"
  echo "[A1] $model_key × $bench × $pool_type (K=$K_arg)"
  mkdir -p "$cell_dir"
  
  PYTHONPATH=. python << PYEOF
import asyncio, json, time, yaml
from pathlib import Path
from openai import AsyncOpenAI
from src.data.arc_challenge import load_arc_challenge
from src.data.gsm8k import load_gsm8k
from src.data.mmlu import load_mmlu_subset
from src.llm.extraction import extract_answer
from src.llm.labeling import label
from src.llm.prompts import format_prompt, NUM_VARIATIONS
from src.protocols.pools import split_pools_by_instance
from src.utils.schema import PanelARecord, validate_records
from src.utils.seeds import derive_seed

cfg = yaml.safe_load(open('configs/panel_a_protocol_a1.yaml'))
bench_cfg = yaml.safe_load(open('configs/benchmarks.yaml'))
loaders = {'arc_challenge': load_arc_challenge, 'gsm8k': load_gsm8k, 'mmlu_subset': load_mmlu_subset}

bench = '$bench'
model_key = '$model_key'
hf_path = '$hf_path'
pool_type = '$pool_type'
K = int('$K_arg')
cell_dir = Path('$cell_dir')
seed = $SEED

items = loaders[bench](bench_cfg[bench], seed)
ids = [it['instance_id'] for it in items]
items_by_id = {it['instance_id']: it for it in items}
est_ids, orc_ids = split_pools_by_instance(ids, float(cfg['oracle_fraction']), seed)
pool_ids = est_ids if pool_type == 'estimation' else orc_ids
row_of = {iid: m for m, iid in enumerate(pool_ids)}

client = AsyncOpenAI(base_url='http://localhost:$port/v1', api_key='EMPTY')
SEMAPHORE = asyncio.Semaphore(256)  # 限制 in-flight 请求数, 避免 server 压垮

async def one_request(instance_id, k):
    async with SEMAPHORE:
        item = items_by_id[instance_id]
        prompt, var_id = format_prompt(item, bench, k % NUM_VARIATIONS)
        gen_seed = derive_seed(seed, bench, model_key, instance_id, k, 'gen') % (2**63)
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=hf_path,
                messages=[{'role': 'user', 'content': prompt}],
                n=1, temperature=float(cfg['temperature']),
                max_tokens=int(cfg['max_tokens']),
                top_p=float(cfg['top_p']),
                seed=gen_seed,
            )
            raw = resp.choices[0].message.content or ''
        except Exception as e:
            raise RuntimeError(f'gen failed for {instance_id} k={k}: {e!r}')
        latency_ms = (time.time() - t0) * 1000.0
        ext, invalid = extract_answer(raw, bench, log_path=cell_dir / 'extraction.jsonl', instance_id=instance_id)
        success = 0 if invalid else int(label(ext, item['gold_answer'], bench))
        return {
            'instance_id': instance_id, 'pool': pool_type, 'm': row_of[instance_id], 'k': k,
            'prompt_variation_id': var_id, 'prompt': prompt, 'raw_completion': raw,
            'extracted_answer': ext, 'gold_answer': str(item['gold_answer']),
            'success_indicator': success, 'invalid_parse': bool(invalid),
            'model': model_key, 'latency_ms': latency_ms, 'seed': gen_seed,
        }

async def main():
    tasks = [one_request(iid, k) for iid in pool_ids for k in range(K)]
    print(f'  submitting {len(tasks)} requests with concurrency 256...')
    results = []
    done = 0
    t_start = time.time()
    # gather in chunks so progress is visible and memory bounded
    CHUNK = 2000
    for i in range(0, len(tasks), CHUNK):
        chunk_results = await asyncio.gather(*tasks[i:i+CHUNK])
        results.extend(chunk_results)
        done += len(chunk_results)
        elapsed = time.time() - t_start
        rate = done / elapsed
        eta = (len(tasks) - done) / rate if rate > 0 else 0
        print(f'  [{done}/{len(tasks)}] {rate:.1f} req/s, ETA {eta/60:.1f} min')
    return results

results = asyncio.run(main())
validated = validate_records(results, PanelARecord)
out_path = cell_dir / f'{pool_type}.jsonl'
with out_path.open('w') as fh:
    for r in validated:
        fh.write(json.dumps(r, default=str) + '\n')
print(f'  wrote {len(validated)} records to {out_path}')
PYEOF
}

# ===== Verify pre-downloaded model weights =====
echo "===== Verifying pre-downloaded model weights ====="
for local_dir in \
  /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  /root/autodl-tmp/models/Qwen2.5-14B-Instruct; do
  if [ ! -f "$local_dir/config.json" ]; then
    echo "MISSING: $local_dir/config.json"
    exit 1
  fi
  echo "[ok] $local_dir"
done

echo '===== SKIPPING Phase 1.1 + Phase 2 (pilot already validated, coverage=1.000) ====='

# ===== Phase 3: Full Panel A =====
echo "===== SKIPPING Phase 3a A1 (already complete: 9 cells × 64000 rows) ====="
echo ""
echo "===== Phase 3b: Protocol A2 (serial model-swap via existing code) ====="
# Protocol A2 用源码里的 protocol_a2.py（它自己管 server lifecycle），
# 但同样改为并发提交。这里 inline 一个并发版本。

K_FULL_A2=$(python -c "import yaml; print(yaml.safe_load(open('configs/panel_a_protocol_a2.yaml'))['K_full'])")
K_REF_A2=$(python -c "import yaml; print(yaml.safe_load(open('configs/panel_a_protocol_a2.yaml'))['K_ref'])")
SEED_A2=48879  # 0xBEEF

PYTHONPATH=. python << 'PYEOF'
import asyncio, json, subprocess, time, yaml, signal, urllib.request, urllib.error
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI
from src.data.arc_challenge import load_arc_challenge
from src.data.gsm8k import load_gsm8k
from src.llm.extraction import extract_answer
from src.llm.labeling import label
from src.llm.prompts import format_prompt, NUM_VARIATIONS
from src.protocols.pools import split_pools_by_instance
from src.utils.schema import PanelARecord, validate_records
from src.utils.seeds import derive_seed, rng_for

cfg = yaml.safe_load(open('configs/panel_a_protocol_a2.yaml'))
bench_cfg = yaml.safe_load(open('configs/benchmarks.yaml'))
models_cfg = yaml.safe_load(open('configs/models.yaml'))
loaders = {'arc_challenge': load_arc_challenge, 'gsm8k': load_gsm8k}
seed = 48879
names = [e['name'] for e in cfg['model_pool']]
weights = np.array([float(e['weight']) for e in cfg['model_pool']])
weights = weights / weights.sum()

def start_server(model_key):
    mcfg = models_cfg[model_key]
    hf_path = mcfg['hf_path']
    vcfg = mcfg['vllm']
    extra = ['--trust-remote-code'] if 'qwen' in model_key.lower() else []
    cmd = ['vllm', 'serve', hf_path,
        '--port', str(vcfg['port']),
        '--gpu-memory-utilization', str(vcfg['gpu_memory_utilization']),
        '--max-model-len', str(vcfg['max_model_len']),
        '--seed', '0', '--enable-prefix-caching',
        '--max-num-seqs', '256', '--max-num-batched-tokens', '4096',
        '--dtype', 'bfloat16',
        '--download-dir', '/root/autodl-tmp/models/_vllm_cache'] + extra
    log_path = Path(f'outputs/logs/vllm_a2_{vcfg["port"]}.log')
    proc = subprocess.Popen(cmd, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    # wait until /v1/models responds
    for i in range(300):
        try:
            urllib.request.urlopen(f'http://localhost:{vcfg["port"]}/v1/models', timeout=2).read()
            print(f'  vLLM {model_key} ready after {i*2}s')
            return proc, vcfg['port'], hf_path
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    raise RuntimeError(f'{model_key} did not become ready in 600s')

def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate(); proc.wait(timeout=10)
    # 杀残留 GPU 进程
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True
        ).strip()
        for pid_s in out.splitlines():
            pid_s = pid_s.strip()
            if pid_s:
                try: subprocess.run(['kill', '-9', pid_s], check=False)
                except Exception: pass
    except Exception: pass
    # 轮询等显存释放 < 1000 MiB, 最多 60 秒
    for i in range(60):
        try:
            mem = int(subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                text=True
            ).strip().split('\n')[0])
            if mem < 1000:
                print(f'  GPU released after {i}s (mem={mem} MiB)')
                return
        except Exception: pass
        time.sleep(1)
    print('  WARNING: GPU mem still high after 60s')

async def one_request(client, hf_path, item, bench, instance_id, k, pool_type, row_of, cell_dir, sem=None):
    prompt, var_id = format_prompt(item, bench, k % NUM_VARIATIONS)
    if True:
        # A2 的 seed key 用 drawn model name
        gen_seed = derive_seed(seed, bench, item['_drawn_model'], instance_id, k, 'gen') % (2**63)
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=hf_path, messages=[{'role': 'user', 'content': prompt}],
            n=1, temperature=float(cfg['temperature']),
            max_tokens=int(cfg['max_tokens']), top_p=float(cfg['top_p']),
            seed=gen_seed,
        )
        raw = resp.choices[0].message.content or ''
        latency_ms = (time.time() - t0) * 1000.0
        ext, invalid = extract_answer(raw, bench, log_path=cell_dir / 'extraction.jsonl', instance_id=instance_id)
        success = 0 if invalid else int(label(ext, item['gold_answer'], bench))
        return {
            'instance_id': instance_id, 'pool': pool_type,
            'm': row_of[instance_id], 'k': k, 'prompt_variation_id': var_id,
            'prompt': prompt, 'raw_completion': raw,
            'extracted_answer': ext, 'gold_answer': str(item['gold_answer']),
            'success_indicator': success, 'invalid_parse': bool(invalid),
            'model': item['_drawn_model'], 'latency_ms': latency_ms, 'seed': gen_seed,
        }

async def run_model_batch(port, hf_path, model_key, assignments, items_by_id, bench, pool_type, row_of, cell_dir):
    SEM = asyncio.Semaphore(256)
    client = AsyncOpenAI(base_url=f'http://localhost:{port}/v1', api_key='EMPTY')
    async def one_request_local(item, iid, k):
        async with SEM:
            return await one_request(client, hf_path, item, bench, iid, k, pool_type, row_of, cell_dir, sem=None)
    tasks = []
    for (iid, k) in assignments:
        item = dict(items_by_id[iid])
        item['_drawn_model'] = model_key
        tasks.append(one_request_local(item, iid, k))
    try:
        return await asyncio.gather(*tasks)
    finally:
        try: await client.close()
        except Exception: pass

for bench in cfg['benchmarks']:
    items = loaders[bench](bench_cfg[bench], seed)
    ids = [it['instance_id'] for it in items]
    items_by_id = {it['instance_id']: it for it in items}
    est_ids, orc_ids = split_pools_by_instance(ids, float(cfg['oracle_fraction']), seed)
    cell_dir = Path(f'outputs/panel_a/A2_{bench}_pool')
    cell_dir.mkdir(parents=True, exist_ok=True)

    for pool_type, pool_ids, K in [('estimation', est_ids, int(cfg['K_full'])),
                                    ('oracle',     orc_ids, int(cfg['K_ref']))]:
        row_of = {iid: m for m, iid in enumerate(pool_ids)}
        # 预先决定每个 (iid, k) 的模型
        assignment = {}
        for iid in pool_ids:
            for k in range(K):
                rng = rng_for(seed, bench, iid, k, 'model_assign')
                assignment[(iid, k)] = str(rng.choice(names, p=weights))
        all_results = {}
        # 检查已完成的模型 checkpoint, 跳过重跑
        for model_key in names:
            ckpt = cell_dir / f'_ckpt_{pool_type}_{model_key}.jsonl'
            if ckpt.exists():
                print(f'[A2] {bench} {pool_type}: loading checkpoint for {model_key}')
                with ckpt.open() as fh:
                    for line in fh:
                        r = json.loads(line)
                        all_results[(r['instance_id'], r['k'])] = r
                continue
            batch = [(iid, k) for (iid, k), mk in assignment.items() if mk == model_key]
            if not batch: continue
            print(f'[A2] {bench} {pool_type}: serving {model_key} for {len(batch)} requests')
            proc, port, hf_path = start_server(model_key)
            try:
                results = asyncio.run(run_model_batch(port, hf_path, model_key, batch,
                                                      items_by_id, bench, pool_type, row_of, cell_dir))
                # 立即落 checkpoint
                with ckpt.open('w') as fh:
                    for r in results:
                        fh.write(json.dumps(r, default=str) + '\n')
                for r in results:
                    all_results[(r['instance_id'], r['k'])] = r
                print(f'[A2] checkpoint saved: {ckpt} ({len(results)} records)')
            finally:
                stop_server(proc)
        # 排序 + 写出
        ordered = [all_results[(iid, k)] for iid in pool_ids for k in range(K)]
        validated = validate_records(ordered, PanelARecord)
        with (cell_dir / f'{pool_type}.jsonl').open('w') as fh:
            for r in validated:
                fh.write(json.dumps(r, default=str) + '\n')
        print(f'[A2] wrote {cell_dir / pool_type}.jsonl ({len(validated)} records)')
PYEOF

echo ""
echo "===== Phase 3c: Synthesize panel_a_summary.json from cell outputs ====="
PYTHONPATH=. python << 'PYEOF'
import json, yaml
from pathlib import Path
import numpy as np
from src.certs.empirical import bonferroni_cell_budget, empirical_certificate
from src.certs.refusal import classify_refusal
from src.utils.provenance import capture

cfg_a1 = yaml.safe_load(open('configs/panel_a_protocol_a1.yaml'))
cfg_a2 = yaml.safe_load(open('configs/panel_a_protocol_a2.yaml'))

cells = []
for bench in cfg_a1['benchmarks']:
    for model in cfg_a1['models']:
        cells.append({'protocol': 'A1', 'benchmark': bench, 'model': model,
                      'K_est': int(cfg_a1['K_full']),
                      'cell_dir': f'outputs/panel_a/A1_{bench}_{model}'})
pool_name = 'pool:' + '+'.join(e['name'] for e in cfg_a2['model_pool'])
for bench in cfg_a2['benchmarks']:
    cells.append({'protocol': 'A2', 'benchmark': bench, 'model': pool_name,
                  'K_est': int(cfg_a2['K_full']),
                  'cell_dir': f'outputs/panel_a/A2_{bench}_pool'})
C = len(cells)
delta_global = float(cfg_a1['delta_global'])
delta_cell = bonferroni_cell_budget(delta_global, C)
N_values = [int(n) for n in cfg_a1['N_values']]

def load_matrix(jsonl_path, K):
    with open(jsonl_path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    M = max(r['m'] for r in recs) + 1
    mat = np.full((M, K), -1, dtype=int)
    for r in recs:
        mat[r['m'], r['k']] = r['success_indicator']
    if (mat < 0).any():
        raise RuntimeError(f'incomplete matrix from {jsonl_path}')
    return mat

cell_rows = []
for cell in cells:
    K_est = cell['K_est']
    cell_dir = Path(cell['cell_dir'])
    est_mat = load_matrix(cell_dir / 'estimation.jsonl', K_est)
    M_est = est_mat.shape[0]
    # 加载 oracle 矩阵, K_ref 取 yaml 配置
    K_ref = int(cfg_a1['K_ref']) if cell['protocol'] == 'A1' else int(cfg_a2['K_ref'])
    orc_mat = load_matrix(cell_dir / 'oracle.jsonl', K_ref)
    M_orc = orc_mat.shape[0]
    from src.analysis.a1_mc_bootstrap import _r_n_mc
    for N in N_values:
        cert = empirical_certificate(est_mat, N, delta_cell, use_BA=True)
        refusal = classify_refusal(cert, epsilon=0.2)
        R_N_MC = _r_n_mc(orc_mat, N)
        cell_rows.append({
            'protocol': cell['protocol'], 'benchmark': cell['benchmark'], 'model': cell['model'],
            'K_est': K_est, 'M_estimation': M_est, 'M_oracle': M_orc, 'N': N,
            'delta_global': delta_global, 'delta_cell': delta_cell, 'C_cells': C,
            'alpha_bar_hat': cert['alpha_bar_hat'], 'F_hat': cert['F_hat'],
            'L_alpha': cert['L_alpha'], 'U_alpha': cert['U_alpha'], 'U_F': cert['U_F'],
            'm_L': cert['m_L'], 'm_beta_L': cert['m_beta_L'],
            'R_N_cert': cert['R_N_cert'], 'Q_N_cert': cert['Q_N_cert'],
            'R_N_BA_cert': cert['R_N_BA_cert'],
            'R_N_MC': R_N_MC,
            'refusal_mode': refusal['mode'], 'refusal_sub_mode': refusal['sub_mode'],
        })

summary = {
    'C': C, 'C_cells': C,
    'delta_global': delta_global, 'delta_cell': delta_cell,
    'N_values': N_values, 'cells': cells, 'cell_certificates': cell_rows,
    'provenance': capture(51966, ['configs/panel_a_protocol_a1.yaml', 'configs/panel_a_protocol_a2.yaml']),
}
Path('outputs/panel_a/panel_a_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(f'wrote outputs/panel_a/panel_a_summary.json with C={C}, delta_cell={delta_cell:.4e}')
PYEOF

# 修正 05_run_analyses.py 期待的 cell_dir 字段映射
# 05 脚本第 192 行写死了 f"{cell['protocol']}_{cell['benchmark']}"
# 但我们的目录是 f"A1_{benchmark}_{model}" / f"A2_{benchmark}_pool"
# 解决：要么改 05 脚本, 要么在 summary 里直接给 cell_dir。
# 上面 summary 已经写了 cell['cell_dir']，需要确认 05 用的字段。
# (如果 05 还是用 protocol_benchmark, 需要手工 patch。下面给 patch。)

echo ""
echo "===== Phase 4: Analyses + Figures + Tables ====="
# 如果 05_run_analyses.py 仍按旧约定读取目录, 需 sed patch:
# 这里假设你已经把 05 脚本里的目录解析改为读取 cell['cell_dir']。
# 若未改, 下面的 Phase 4 会因目录找不到而报错。

PYTHONPATH=. python scripts/05_run_analyses.py \
  --panel_a_dir outputs/panel_a --output_dir outputs/analyses

PYTHONPATH=. python scripts/06_render_figures.py \
  --analyses_dir outputs/analyses --panel_a_dir outputs/panel_a \
  --output_dir outputs/figures

PYTHONPATH=. python scripts/07_build_tables.py \
  --analyses_dir outputs/analyses \
  --output_dir outputs/analyses/tables

echo ""
echo "===== ALL PHASES COMPLETE ====="
echo "Key outputs:"
echo "  outputs/figures/analysis7_stacked_bar.{png,pdf}  ← 论文核心图"
echo "  outputs/figures/design_space.{png,pdf}"
echo "  outputs/figures/budget_curves.{png,pdf}"
echo "  outputs/analyses/tables/*.{tex,md}"
