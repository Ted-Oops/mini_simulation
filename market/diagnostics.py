import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from market.estimators import paired_bootstrap
from market.statistics import hill
from market.storage import write_json


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
    print(f'统计口径核对完成：{root}',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    audit(parser.parse_args().directory.resolve())
