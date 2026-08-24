from pathlib import Path
import json
import statistics
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, TRADING_CONFIG

import pandas as pd

pairs = TRADING_CONFIG['pairs']
tfs = ['1h', '4h', '1d']
thresholds = {'1h': 90000, '4h': 20000, '1d': 3600}

missing = []
partial = []
rows = []
for symbol in pairs:
    for tf in tfs:
        path = DATA_DIR / f'{symbol}_{tf}_history.parquet'
        if not path.exists():
            missing.append((symbol, tf))
            rows.append({'symbol': symbol, 'timeframe': tf, 'exists': False, 'rows': 0})
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            missing.append((symbol, tf))
            rows.append({'symbol': symbol, 'timeframe': tf, 'exists': False, 'rows': 0, 'error': str(e)})
            continue
        if len(df) == 0:
            missing.append((symbol, tf))
            rows.append({'symbol': symbol, 'timeframe': tf, 'exists': True, 'rows': 0})
            continue
        if 'time' in df.columns:
            ts = pd.to_datetime(df['time'])
        else:
            ts = pd.to_datetime(df.index)
        row_count = len(df)
        start = str(ts.min())
        end = str(ts.max())
        partial_flag = row_count < thresholds[tf]
        if partial_flag:
            partial.append((symbol, tf, row_count, start, end))
        rows.append({'symbol': symbol, 'timeframe': tf, 'exists': True, 'rows': row_count, 'start': start, 'end': end, 'partial': partial_flag})

meta_dir = Path('ai_engine/models/saved')
meta_files = sorted(meta_dir.glob('*.metadata.json'))
accs = []
latest = {}
for p in meta_files:
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    sym = data.get('symbol')
    tf = data.get('timeframe')
    acc = data.get('validation', {}).get('candidate_model_accuracy')
    ver = data.get('version')
    if sym and tf and acc is not None:
        key = (sym, tf)
        if key not in latest or (ver is not None and str(ver) > str(latest[key][0])):
            latest[key] = (ver, acc, p.name)
        accs.append((sym, tf, acc, ver, p.name))

acc_values = [a for (_, _, a, _, _) in accs]
counts = {}
for _, tf, _, _, _ in accs:
    counts[tf] = counts.get(tf, 0) + 1

report = {
    'data_status': {
        'total_needed': len(pairs) * len(tfs),
        'missing_count': len(missing),
        'partial_count': len(partial),
        'complete_count': len(pairs) * len(tfs) - len(missing) - len(partial),
        'missing_samples': missing[:40],
        'partial_samples': partial[:40],
    },
    'model_status': {
        'total_metadata_files': len(meta_files),
        'accuracy_entries': len(accs),
        'latest_model_pairs': len(latest),
        'max_accuracy': max(acc_values) if acc_values else None,
        'min_accuracy': min(acc_values) if acc_values else None,
        'mean_accuracy': statistics.mean(acc_values) if acc_values else None,
        'median_accuracy': statistics.median(acc_values) if acc_values else None,
        'accuracy_counts_by_timeframe': counts,
        'latest_sample': sorted([(k[0], k[1], v[1], v[0], v[2]) for k, v in latest.items()])[:50],
    },
}

log_path = Path('logs/auto_retrain.log')
if log_path.exists():
    text = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    tail = [line for line in text[-120:] if any(tok in line for tok in ['starting download pass', 'starting training pass', 'EXIT', 'Accuracy', 'Precision', 'completed successfully', 'failed', 'downloaded', 'trailing failures'])]
    report['auto_retrain_log'] = tail
    report['auto_retrain_log_last_modified'] = log_path.stat().st_mtime
else:
    report['auto_retrain_log'] = None

print(json.dumps(report, indent=2))
