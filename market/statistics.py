import math

import numpy as np

from market.estimators import modified_rs


def hill(values, fraction=0.05):
    
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    k = int(len(values) * fraction)
    positive = np.sort(values[values > 1e-14])[::-1]
    if k < 20 or k >= len(positive):
        return None
    gamma = float(np.mean(np.log(positive[:k] / positive[k])))
    return 1 / gamma if gamma > 1e-12 else None


def acf(values, lags=800):
    centered = np.asarray(values, dtype=float) - np.mean(values)
    size = 1 << (2 * len(centered) - 1).bit_length()
    transformed = np.fft.rfft(centered, size)
    covariances = np.fft.irfft(transformed * transformed.conjugate(), size)[:min(lags + 1, len(centered))]
    if covariances[0] <= 1e-28:
        return [None] * lags
    result = list((covariances[1:] / covariances[0]).astype(float))
    return result + [None] * (lags - len(result))


def mean_finite(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(np.mean(valid)) if len(valid) else None


def evaluate(data, burn=0):
    arrays = {name: values[burn:] for name, values in data.items()}
    returns = arrays['return']
    absolute = np.abs(returns)
    variance = float(np.var(returns))
    absolute_acf = acf(absolute)
    beta = {}
    for window in (100, 200, 400, 600, 800):
        
        values = [modified_rs(absolute[start:start + window], q=20)
                  for start in range(0, len(absolute) - window + 1, window)]
        beta[str(window)] = mean_finite([x for x in values if x is not None])
    market = arrays['market_order'] > 0
    buys, sells = market & (arrays['side'] == 1), market & (arrays['side'] == -1)
    error = arrays['price'] / arrays['fundamental'] - 1
    result = {
        'observations': len(returns), 'return_volatility': math.sqrt(variance),
        'fundamental_error': float(np.mean(np.abs(error))), 'signed_fundamental_error': float(np.mean(error)),
        'hill_absolute': hill(absolute), 'hill_positive': hill(returns), 'hill_negative': hill(-returns),
        'hill_positive_side_fraction': hill(returns[returns > 1e-14]),
        'hill_negative_side_fraction': hill(-returns[returns < -1e-14]),
        'excess_kurtosis': float(np.mean((returns - returns.mean()) ** 4) / variance ** 2 - 3) if variance else None,
        'zero_return_ratio': float(np.mean(absolute < 1e-14)),
        'abs_acf1': absolute_acf[0], 'abs_acf100': absolute_acf[99],
        'abs_acf_100_800': mean_finite([x for x in absolute_acf[99:] if x is not None]),
        'absolute_acf': absolute_acf, 'return_acf': acf(returns), 'lo_beta': beta,
        'hill_bid_gap': hill(arrays['bid_gap']), 'hill_ask_gap': hill(arrays['ask_gap']),
        'hill_gap': hill(np.concatenate((arrays['bid_gap'], arrays['ask_gap']))),
        'hill_placement': hill(arrays['placement_best']),
        'placement_mid': mean_finite(arrays['placement_mid']),
        'mean_spread': mean_finite(arrays['spread']), 'mean_active_orders': float(np.mean(arrays['active_orders'])),
        'two_sided_ratio': float(np.mean(arrays['two_sided'])),
        'volume_per_event': float(np.mean(arrays['executed_volume'])),
        'market_buy_count': int(buys.sum()), 'market_sell_count': int(sells.sum()),
        'market_buy_mean_size': mean_finite(arrays['submitted_volume'][buys]),
        'market_sell_mean_size': mean_finite(arrays['submitted_volume'][sells]),
        'market_volume_bins': [int((market & condition).sum()) for condition in
                               (arrays['submitted_volume'] < 15,
                                (arrays['submitted_volume'] >= 15) & (arrays['submitted_volume'] <= 30),
                                arrays['submitted_volume'] > 30)],
    }
    return result
