"""异质交易规则限价订单簿：模拟市场并生成论文结果核对图表。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np

from market.config import MarketConfig
from market.engine import simulate
from market.statistics import evaluate
from market.storage import snapshot, write_json


SCENARIOS = {'noise': (0.0, 0.0), 'fundamental': (10.0, 0.0),
             'chart12': (10.0, 1.2), 'chart20': (10.0, 2.0), 'chart28': (10.0, 2.8),
             'fund05': (5.0, 0.0), 'fund15': (15.0, 0.0), 'fund20': (20.0, 0.0),
             'chart04': (10.0, 0.4), 'chart08': (10.0, 0.8), 'chart16': (10.0, 1.6),
             'chart24': (10.0, 2.4), 'chart30': (10.0, 3.0)}


def worker(job):
    root, name, parameters = job
    start = time.perf_counter()
    config = MarketConfig(**parameters)
    folder = Path(root) / 'runs' / f'{name}_seed{config.seed}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'config.json', parameters)
    try:
        data, metadata = simulate(config)
        np.savez_compressed(folder / 'events.npz', **data)
        metrics = evaluate(data, config.initialization_steps + config.burn_in)
        metadata['metrics'] = metrics
        metadata['post_10pct_metrics'] = evaluate(data, config.initialization_steps + max(config.burn_in, config.steps // 10))
        metadata['elapsed_seconds'] = time.perf_counter() - start
        write_json(folder / 'summary.json', metadata)
        return {'scenario': name, 'seed': config.seed, 'status': 'success',
                'metrics': metrics, 'elapsed_seconds': metadata['elapsed_seconds'],
                'checks': metadata['checks'], 'variance_floor_events': metadata['variance_floor_events']}
    except Exception as exc:
        failure = {'scenario': name, 'seed': config.seed, 'status': 'failed',
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'elapsed_seconds': time.perf_counter() - start}
        write_json(folder / 'failure.json', failure)
        return failure


def run(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='运行：python main.py run [参数]；重建已有结果图表：python main.py report [结果目录]。')
    parser.add_argument('--scenarios', default='noise,fundamental,chart12,chart20,chart28')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--agents', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--burn-in', type=int, default=0)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--noise-interpretation', choices=['std', 'variance'], default='std')
    parser.add_argument('--fundamental-horizon', type=float, default=200.0)
    parser.add_argument('--fundamental-horizon-mode', choices=['fixed', 'individual'], default='fixed')
    parser.add_argument('--grid-rule', choices=['sample_then_round', 'discrete_interval'], default='sample_then_round')
    parser.add_argument('--noise-parameter', type=float, default=1e-4)
    parser.add_argument('--initialization-steps', type=int, default=0)
    parser.add_argument('--trait-weights', choices=['raw', 'normalized'], default='raw')
    parser.add_argument('--order-policy', choices=['multiple', 'replace'], default='multiple')
    parser.add_argument('--warmup', choices=['noise', 'fundamental', 'flat'], default='noise')
    parser.add_argument('--fundamental-clock', choices=['event', 'population'], default='event')
    parser.add_argument('--quantity-mode', choices=['continuous', 'integer'], default='continuous')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-report', action='store_true', help='只保存模拟数据，稍后通过 report 命令生成图表')
    args = parser.parse_args(argv)
    try:
        seeds = [int(x) for x in args.seeds.split(',')]
        scenarios = args.scenarios.split(',')
        if not seeds or len(seeds) != len(set(seeds)) or len(scenarios) != len(set(scenarios)):
            raise ValueError('种子及情形必须非空且不能重复')
        if any(name not in SCENARIOS for name in scenarios):
            raise ValueError('情形名称不存在')
        if args.workers < 1:
            raise ValueError('工作进程数必须为正')
        base = MarketConfig(agents=args.agents, steps=args.steps, burn_in=args.burn_in,
                           noise_interpretation=args.noise_interpretation, fundamental_horizon=args.fundamental_horizon,
                           fundamental_horizon_mode=args.fundamental_horizon_mode,
                           grid_rule=args.grid_rule, noise_parameter=args.noise_parameter,
                           initialization_steps=args.initialization_steps,
                           trait_weights=args.trait_weights,
                           order_policy=args.order_policy, warmup=args.warmup,
                           fundamental_clock=args.fundamental_clock, quantity_mode=args.quantity_mode)
    except ValueError as exc:
        parser.error(str(exc))
    root = (args.output or Path('results') / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'base_config': asdict(base), 'scenarios': {name: SCENARIOS[name] for name in scenarios},
                'seeds': seeds, 'source_sha256': snapshot(root), 'python': platform.python_version(),
                'numpy': np.__version__, 'created_at': datetime.now().astimezone().isoformat(), 'complete': False}
    write_json(root / 'manifest.json', manifest)
    jobs = []
    for name in scenarios:
        wf, wc = SCENARIOS[name]
        for seed in seeds:
            jobs.append((str(root), name, asdict(replace(base, seed=seed, fundamental_weight=wf, chartist_weight=wc))))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            write_json(root / 'results.json', sorted(results, key=lambda r: (r['scenario'], r['seed'])))
            metrics = result.get('metrics', {})
            print(f"复现进度 {len(results)}/{len(jobs)}：{result['scenario']}，种子 {result['seed']}，"
                  f"状态 {result['status']}，价值偏离 {metrics.get('fundamental_error')}，"
                  f"尾指数 {metrics.get('hill_absolute')}，耗时 {result['elapsed_seconds']:.1f} 秒", flush=True)
    manifest['complete'] = True
    manifest['successful_runs'] = sum(r['status'] == 'success' for r in results)
    manifest['failed_runs'] = sum(r['status'] == 'failed' for r in results)
    write_json(root / 'manifest.json', manifest)
    print(f'复现目录：{root}', flush=True)
    if not args.no_report and manifest['successful_runs']:
        report(root)
    return 2 if manifest['failed_runs'] else 0


def report(root):
    from market.plotting import generate
    from market.diagnostics import audit
    generate(root)
    audit(root)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'report':
        parser = argparse.ArgumentParser(description='从已有实验数据重新生成图表、中文报告和统计核验。')
        parser.add_argument('directory', type=Path, nargs='?', default=Path('results/baseline'))
        report(parser.parse_args(arguments[1:]).directory.resolve())
        return 0
    if arguments and arguments[0] == 'run':
        arguments.pop(0)
    return run(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
