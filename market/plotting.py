"""从已保存的本地事件生成论文图形对应项；不修改或重跑市场。"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from market.fonts import configure_chinese_font
from market.estimators import paired_bootstrap
from market.storage import write_json


LABELS = {'noise': '纯噪声', 'fundamental': '基本面与噪声', 'chart12': '加入趋势（1.2）',
          'chart20': '加入趋势（2.0）', 'chart28': '加入趋势（2.8）'}
COLORS = {'noise': '#202020', 'fundamental': '#b91c1c', 'chart12': '#15803d', 'chart20': '#2563eb', 'chart28': '#7c3aed'}
METRICS = {'fundamental_error': '基本价值相对偏离', 'signed_fundamental_error': '有符号价值偏离',
           'return_volatility': '收益标准差', 'hill_positive': '右尾指数', 'hill_negative': '左尾指数',
           'hill_absolute': '绝对收益尾指数', 'abs_acf1': '绝对收益一阶自相关',
           'hill_positive_side_fraction': '右尾指数（单侧非零5%）',
           'hill_negative_side_fraction': '左尾指数（单侧非零5%）',
           'abs_acf_100_800': '绝对收益滞后100至800均值', 'zero_return_ratio': '零收益比例',
           'volume_per_event': '每事件成交股数', 'two_sided_ratio': '双边盘口比例',
           'hill_gap': '首档缺口尾指数', 'hill_placement': '来单距同侧最优价尾指数',
           'placement_mid': '来单距中间价均值'}


def finite(values):
    return [v for v in values if v is not None and np.isfinite(v)]


def estimate(values):
    return paired_bootstrap(finite(values))


def fmt(value):
    return f'{value:.6g}' if value is not None else '不足'


def interval(item):
    return f"[{fmt(item['ci_low'])}, {fmt(item['ci_high'])}]" if item['ci_low'] is not None else '种子不足'


def ddf(ax, values, label, color, normalize=True):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    positive = np.sort(values[values > 1e-14])
    if not len(positive):
        return
    indices = np.unique(np.linspace(0, len(positive) - 1, min(2000, len(positive))).astype(int))
    probability = (len(positive) - indices) / (len(values) if normalize else 1)
    ax.loglog(positive[indices], probability, label=label, color=color)


def save(fig, root, name):
    fig.savefig(root/name, dpi=160)
    plt.close(fig)


def generate(root):
    print(f'中文绘图字体：{configure_chinese_font()}', flush=True)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    results = json.loads((root/'results.json').read_text(encoding='utf-8'))
    successful = [r for r in results if r['status'] == 'success']
    groups = {name: sorted([r for r in successful if r['scenario'] == name], key=lambda r:r['seed'])
              for name in sorted({r['scenario'] for r in successful})}
    estimates = {name: {metric: estimate([r['metrics'].get(metric) for r in members]) for metric in METRICS}
                 for name, members in groups.items()}
    contrasts=[]
    for comparison, reference in [('fundamental','noise'), ('chart12','fundamental'), ('chart20','fundamental'), ('chart28','fundamental')]:
        if reference not in groups or comparison not in groups:
            continue
        base = {r['seed']:r for r in groups[reference]}
        for metric in METRICS:
            differences=[r['metrics'][metric]-base[r['seed']]['metrics'][metric] for r in groups[comparison]
                         if r['seed'] in base and r['metrics'].get(metric) is not None
                         and base[r['seed']]['metrics'].get(metric) is not None]
            contrasts.append({'comparison':comparison,'reference':reference,'metric':metric,**estimate(differences)})
    write_json(root/'group_estimates.json', estimates)
    write_json(root/'paired_comparisons.json', contrasts)
    with (root/'逐种子统计.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['情形','种子',*METRICS.values()])
        for r in sorted(successful,key=lambda r:(r['scenario'],r['seed'])):
            writer.writerow([LABELS.get(r['scenario'],r['scenario']),r['seed'],*[r['metrics'].get(k) for k in METRICS]])
    lines=['# 原文复现结果核对','',
           '**当前报告逐项展示匹配与偏差，不能仅凭程序成功或部分曲线相似宣布论文已完整复现。**','',
           f"运行完成标记：{manifest.get('complete',False)}；已保存成功组：{len(successful)}；失败组：{len(results)-len(successful)}。",
           f"基准配置：`{json.dumps(manifest['base_config'],ensure_ascii=False)}`。",'',
           'σ1、σ2 控制同一人口中每人的混合权重分布；没有纯类型人数替代，没有投机成分、LLM 或外部补单。',
           '所有图形均为当前实现与明确假设下的输出。原文未明确的 τ_f、噪声口径、旧单处理和初始化见项目复现协议。','',
           '## 分组均值与跨种子区间','',
           '| 情形 | 指标 | 均值 | 95%自助法区间 | 种子数 |','|---|---|---:|---|---:|']
    for name,members in groups.items():
        for metric in METRICS:
            item=estimates[name][metric]
            lines.append(f"| {LABELS.get(name,name)} | {METRICS[metric]} | {fmt(item['mean'])} | {interval(item)} | {item['n']} |")
    lines += ['','## 配对差异','',
              '同一随机种子共享基本价值随机数流；差异为比较组减参考组。未做多重比较校正。','',
              '| 比较 | 指标 | 差异 | 95%区间 | 配对数 |','|---|---|---:|---|---:|']
    for item in contrasts:
        lines.append(f"| {LABELS.get(item['comparison'],item['comparison'])} 相对 {LABELS.get(item['reference'],item['reference'])} | {METRICS[item['metric']]} | {fmt(item['mean'])} | {interval(item)} | {item['n']} |")
    lines += ['','## 论文目标与核对口径','',
              '- 图2：基本面加入后应明显跟踪且接近基本价值；均值偏离降低仅支持改善，不能自动等同于图中紧密跟踪。',
              '- 图3—4：论文无趋势时左右尾指数约3.5至4.5；趋势增加后指数下降到3以下。请逐组比较，不能把所有厚尾都算作趋势效应。',
              '- 图5：同时检查绝对收益自相关与 q=20 的窗口修正 R/S；不以单个 β 值代替正式长期记忆检验。',
              '- 图6—10：检查买卖市价单、挂单距离、首档缺口与时间平均盘口。分布曲线相似不能替代分布拟合检验。',
              '- 图11：检查不同来单量下的收益尾和跨趋势权重的缺口—收益尾指数关系，不能据相关图声称已识别因果。','',
              '## 数据与数值边界','',
              '- `events.npz` 保存完整逐事件字段，包含初始化事件（若配置了初始化）；主统计从初始化结束和指定剔除期后计算。',
              '- `summary.json` 还保存剔除主测量样本前10%后的统计，用来检查启动非平稳性，不能事后只保留有利版本。',
              '- 原文全样本5%尾阈值为当前主口径；另保存单侧样本5%结果。缺少足够正尾观测时返回空值。',
              '- 买卖量条件分组使用主动提交的需求数量；成交量另记，预算绑定也单独记录。',
              '- 时间平均盘口按配置的事件间隔采样，空盘口包含在分母；需收敛检查后才能等同于每事件平均。',
              '- 路径、分布图只展示最小种子的示例；尾指数与相关图显示跨种子的均值和区间。',
              '- 资金、库存守恒通过只表示市场实现满足账户条件，不保证原文现象已经复现。','']
    (root/'复现核对报告.md').write_text('\n'.join(lines),encoding='utf-8')
    examples={}
    summaries={}
    for name,members in groups.items():
        folder=root/'runs'/f"{name}_seed{members[0]['seed']}"
        summaries[name]=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        start=manifest['base_config'].get('initialization_steps',0)+manifest['base_config']['burn_in']
        with np.load(folder/'events.npz') as archive:
            examples[name]={k:archive[k][start:] for k in archive.files if k in
                            ('price','fundamental','return','submitted_volume','side','market_order','placement_mid','opposite_gap')}
    core=[name for name in ('noise','fundamental','chart12') if name in groups]
    if core:
        fig,axes=plt.subplots(2,len(core),figsize=(5*len(core),7),squeeze=False,constrained_layout=True)
        for col,name in enumerate(core):
            d=examples[name];idx=np.unique(np.linspace(0,len(d['price'])-1,min(4000,len(d['price']))).astype(int))
            axes[0,col].plot(idx,d['price'][idx],color='black',label='市场价')
            axes[0,col].plot(idx,d['fundamental'][idx],color='#dc2626',label='基本价值')
            axes[0,col].set(title=LABELS[name],xlabel='测量事件',ylabel='价格');axes[0,col].legend()
            axes[0,col].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value/10000:g}'))
            axes[0,col].set_xlabel('测量事件（万）')
            ddf(axes[1,col],np.abs(d['return']),LABELS[name],COLORS[name])
            axes[1,col].set(xlabel='绝对对数收益',ylabel='超越概率')
        fig.suptitle('图2—3对应：价格与收益（最小种子，全测量区间）')
        save(fig,root,'图2_3_价格与收益.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for name in core:
            members=groups[name]
            curve=np.nanmean([[np.nan if x is None else x for x in r['metrics']['absolute_acf']] for r in members],axis=0)
            axes[0].plot(range(1,801),curve,label=LABELS[name],color=COLORS[name])
            windows=[100,200,400,600,800]
            means=[estimate([r['metrics']['lo_beta'][str(w)] for r in members])['mean'] for w in windows]
            axes[1].plot(windows,means,'o-',label=LABELS[name],color=COLORS[name])
        axes[0].set(title='绝对收益自相关：跨种子均值',xlabel='滞后事件数',ylabel='自相关')
        axes[1].set(title='修正R/S：窗口均值，q=20',xlabel='窗口长度',ylabel='β')
        axes[1].axhline(.5,color='gray',ls='--');axes[0].legend();axes[1].legend()
        save(fig,root,'图5_波动相关与修正极差.png')
        for field,title,filename,market in [('submitted_volume','市价单需求数量','图6_市价单数量.png',True),
                                            ('placement_mid','挂单距中间价（价格单位）','图7_限价单距离.png',False)]:
            fig,axes=plt.subplots(1,len(core),figsize=(5*len(core),4),squeeze=False,constrained_layout=True)
            for col,name in enumerate(core):
                d=examples[name]
                for side,color,label in [(1,'black','买入'),(-1,'#dc2626','卖出')]:
                    mask=d['side']==side
                    if market:mask &= d['market_order']>0
                    ddf(axes[0,col],d[field][mask],label,color,normalize=not market)
                axes[0,col].set(title=LABELS[name],xlabel=title,ylabel='超越数量' if market else '超越概率')
                axes[0,col].legend()
            save(fig,root,filename)
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for name in core:
            profile=summaries[name]['book_profile'];combined=Counter()
            for side in profile.values():
                for distance,volume in side.items():combined[int(distance)]+=volume
            # 对数箱平均每格挂单量；明确以价位格计距离，避免稀疏格直接连线。
            bins={}
            for distance,volume in combined.items():
                if distance>0:
                    key=math.floor(math.log10(distance)*20);bins[key]=bins.get(key,0)+volume
            x=[];y=[]
            for key,volume in sorted(bins.items()):
                lo,hi=10**(key/20),10**((key+1)/20)
                depth=volume/max(1,hi-lo)
                if depth>0:
                    x.append((key+.5)/20);y.append(math.log10(depth))
            ax.plot(x,y,label=LABELS[name],color=COLORS[name])
        ax.set(title='图10对应：平均盘口（最小种子、完整对数范围）',xlabel='log₁₀ 距中间价（最小价位）',ylabel='log₁₀ 每价位平均挂单股数');ax.legend()
        save(fig,root,'图10_平均盘口.png')
    chart_names=sorted([name for name in groups if manifest['scenarios'][name][0]==10],key=lambda n:manifest['scenarios'][n][1])
    if len(chart_names)>1:
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
        for ax,fields,title in [(axes[0],['hill_negative_side_fraction','hill_positive_side_fraction'],'图4对应：左右收益尾指数（单侧5%）'),
                                 (axes[1],['hill_gap'],'图8对应：首档缺口尾指数'),
                                 (axes[2],['hill_placement'],'图8对应：挂单距离尾指数')]:
            for metric in fields:
                names=[name for name in chart_names if estimates[name][metric]['mean'] is not None]
                means=np.array([estimates[n][metric]['mean'] for n in names])
                lows=np.array([estimates[n][metric]['ci_low'] if estimates[n][metric]['ci_low'] is not None else estimates[n][metric]['mean'] for n in names])
                highs=np.array([estimates[n][metric]['ci_high'] if estimates[n][metric]['ci_high'] is not None else estimates[n][metric]['mean'] for n in names])
                ax.errorbar([manifest['scenarios'][n][1] for n in names],means,yerr=[means-lows,highs-means],fmt='o-',capsize=3,label=METRICS[metric])
            ax.set(title=title,xlabel='趋势权重分布均值 σ₂',ylabel='希尔尾指数');ax.legend()
        save(fig,root,'图4_8_尾指数扫描.png')
    if 'chart20' in examples:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        d=examples['chart20'];v=d['submitted_volume'];market=d['market_order']>0
        for condition,label,color in [(v<15,'小于15股','black'),((v>=15)&(v<=30),'15至30股','#dc2626'),(v>30,'超过30股','#15803d')]:
            ddf(axes[0],np.abs(d['return'][market&condition]),label,color)
        axes[0].set(title='来单数量条件下的收益尾（最小种子）',xlabel='绝对对数收益',ylabel='条件超越概率');axes[0].legend()
        x=[estimates[n]['hill_absolute']['mean'] for n in chart_names];y=[estimates[n]['hill_gap']['mean'] for n in chart_names]
        axes[1].scatter(x,y)
        for name,a,b in zip(chart_names,x,y):
            if a is not None and b is not None:axes[1].annotate(str(manifest['scenarios'][name][1]),(a,b))
        axes[1].set(title='跨趋势权重的尾指数对应（跨种子均值）',xlabel='绝对收益尾指数',ylabel='首档缺口尾指数')
        save(fig,root,'图11_数量条件与缺口对应.png')
    print(f'中文报告与图表：{root}',flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    generate(parser.parse_args().directory.resolve())
