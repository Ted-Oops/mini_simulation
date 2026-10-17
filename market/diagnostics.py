"""复核零收益、启动非平稳性和希尔阈值口径；全部来自已保存路径。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from market.storage import write_json
from market.statistics import hill
from market.estimators import paired_bootstrap


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise SystemExit('批次尚未完成，完整核验将在全部运行结束后进行。')
    results=json.loads((root/'results.json').read_text(encoding='utf-8'))
    records=[]
    for index,result in enumerate(results,start=1):
        if result['status']!='success':
            continue
        folder=root/'runs'/f"{result['scenario']}_seed{result['seed']}"
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        offset=config.get('initialization_steps',0)
        with np.load(folder/'events.npz') as source:
            ret=source['return'][offset:];price=source['price'][offset:];value=source['fundamental'][offset:]
        for label,fraction in [('full',0),('post10',.1),('post50',.5),('last10',.9)]:
            start=max(config['burn_in'],int(len(ret)*fraction))
            r=ret[start:];p=price[start:];v=value[start:]
            positive=r[r>1e-14];negative=-r[r<-1e-14];nonzero=np.abs(r[np.abs(r)>1e-14])
            records.append({'scenario':result['scenario'],'seed':result['seed'],'phase':label,
                            'observations':len(r),'positive_observations':len(positive),'negative_observations':len(negative),
                            'fundamental_error':float(np.mean(np.abs(p/v-1))),
                            'signed_error':float(np.mean(p/v-1)), 'return_volatility':float(np.std(r)),
                            'zero_return_ratio':float(np.mean(np.abs(r)<=1e-14)),
                            'hill_positive_full_fraction':hill(r),'hill_negative_full_fraction':hill(-r),
                            'hill_positive_side_fraction':hill(positive),'hill_negative_side_fraction':hill(negative),
                            'hill_absolute_nonzero_fraction':hill(nonzero)})
        if index%25==0:
            print(f'路径口径核验 {index}/{len(results)}',flush=True)
    write_json(root/'path_audit.json',records)
    code=Path(__file__).read_bytes()
    (root/'analysis_source.py').write_bytes(code)
    estimates={}
    fields=['fundamental_error','return_volatility','zero_return_ratio',
            'hill_positive_full_fraction','hill_negative_full_fraction',
            'hill_positive_side_fraction','hill_negative_side_fraction','hill_absolute_nonzero_fraction']
    for scenario in sorted({r['scenario'] for r in records}):
        estimates[scenario]={}
        for phase in ['full','post10','post50','last10']:
            members=[r for r in records if r['scenario']==scenario and r['phase']==phase]
            estimates[scenario][phase]={field:paired_bootstrap([r[field] for r in members]) for field in fields}
    write_json(root/'path_audit_estimates.json',{'analysis_sha256':hashlib.sha256(code).hexdigest(),'estimates':estimates})
    lines=['# 零收益与启动阶段的口径核对','',
           '这里不改变市场路径，仅检查统计定义。所有阶段都报告，不能事后只选择匹配论文的一段。',
           'full=完整测量样本；post10=剔除前10%；post50=后半段；last10=最后10%。',
           '希尔“全样本5%”包含零收益对阈值个数的影响；“单侧5%”先取正收益或负收益绝对值，再取各侧最高5%。','',
           '| 情形 | 阶段 | 价值偏离 | 零收益比例 | 右尾：全样本5% | 右尾：单侧5% | 左尾：单侧5% |','|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):return f'{x:.5g}' if x is not None else '不足'
    for scenario,phases in estimates.items():
        for phase,metrics in phases.items():
            values=[metrics[field]['mean'] for field in ['fundamental_error','zero_return_ratio','hill_positive_full_fraction','hill_positive_side_fraction','hill_negative_side_fraction']]
            lines.append(f'| {scenario} | {phase} | '+' | '.join(fmt(x) for x in values)+' |')
    lines += ['','完整区间见 `path_audit_estimates.json`，逐种子数据见 `path_audit.json`。',
              '单侧尾指数接近论文不表示价格幅度、趋势效应、盘口或其他结果已经匹配。',
              '启动过程造成的非平稳性可能让全样本厚尾估计混入不同波动状态。需用独立样本核验预先定义的剔除规则。','']
    (root/'统计口径核对.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'统计口径核对完成：{root}',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    audit(parser.parse_args().directory.resolve())
