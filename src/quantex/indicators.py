from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def _as_float_array(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Indicator inputs must be one-dimensional")
    return array


def _validate_same_length(*arrays: np.ndarray) -> None:
    lengths = {len(array) for array in arrays}
    if len(lengths) > 1:
        raise ValueError("Indicator inputs must have the same length")


def _validate_period(period: int, *, minimum: int = 1) -> None:
    if period < minimum:
        raise ValueError(f"period must be >= {minimum}")


def _nan_array(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=np.float64)


def _rolling_sum(values: np.ndarray, period: int) -> np.ndarray:
    _validate_period(period)
    result = _nan_array(len(values))
    if len(values) < period:
        return result
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    result[period - 1 :] = cumsum[period:] - cumsum[:-period]
    return result


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    return _rolling_sum(values, period) / period


def _rolling_std(values: np.ndarray, period: int, *, ddof: int = 0) -> np.ndarray:
    _validate_period(period, minimum=ddof + 1)
    result = _nan_array(len(values))
    if len(values) < period:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, period)
    result[period - 1 :] = np.std(windows, axis=1, ddof=ddof)
    return result


def _rolling_min(values: np.ndarray, period: int) -> np.ndarray:
    _validate_period(period)
    result = _nan_array(len(values))
    if len(values) < period:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, period)
    result[period - 1 :] = np.min(windows, axis=1)
    return result


def _rolling_max(values: np.ndarray, period: int) -> np.ndarray:
    _validate_period(period)
    result = _nan_array(len(values))
    if len(values) < period:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, period)
    result[period - 1 :] = np.max(windows, axis=1)
    return result


def _ema_from_array(values: np.ndarray, period: int) -> np.ndarray:
    _validate_period(period)
    result = _nan_array(len(values))
    if len(values) < period:
        return result
    alpha = 2.0 / (period + 1.0)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def _wilder_smoothing(values: np.ndarray, period: int) -> np.ndarray:
    _validate_period(period)
    result = _nan_array(len(values))
    if len(values) < period:
        return result
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = ((result[i - 1] * (period - 1)) + values[i]) / period
    return result


def sma(values: ArrayLike, period: int) -> np.ndarray:
    return _rolling_mean(_as_float_array(values), period)


def ema(values: ArrayLike, period: int) -> np.ndarray:
    return _ema_from_array(_as_float_array(values), period)


def wma(values: ArrayLike, period: int) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(period)
    result = _nan_array(len(array))
    if len(array) < period:
        return result
    weights = np.arange(1, period + 1, dtype=np.float64)
    denominator = weights.sum()
    windows = np.lib.stride_tricks.sliding_window_view(array, period)
    result[period - 1 :] = np.sum(windows * weights, axis=1) / denominator
    return result


def dema(values: ArrayLike, period: int) -> np.ndarray:
    first = ema(values, period)
    first_valid = np.where(~np.isnan(first))[0]
    result = _nan_array(len(first))
    if len(first_valid) < period:
        return result
    second = _ema_from_array(first[first_valid], period)
    second_start = first_valid[0] + period - 1
    aligned_second = _nan_array(len(first))
    aligned_second[second_start:] = second[period - 1 :]
    result[:] = 2.0 * first - aligned_second
    result[:second_start] = np.nan
    return result


def tema(values: ArrayLike, period: int) -> np.ndarray:
    first = ema(values, period)
    first_valid = np.where(~np.isnan(first))[0]
    result = _nan_array(len(first))
    if len(first_valid) < period:
        return result
    second = _ema_from_array(first[first_valid], period)
    second_start = first_valid[0] + period - 1
    second_valid = np.where(~np.isnan(second))[0]
    if len(second_valid) < period:
        return result
    third = _ema_from_array(second[second_valid], period)
    third_start = second_start + second_valid[0] + period - 1
    aligned_second = _nan_array(len(first))
    aligned_third = _nan_array(len(first))
    aligned_second[second_start:] = second[period - 1 :]
    aligned_third[third_start:] = third[period - 1 :]
    result[:] = (3.0 * first) - (3.0 * aligned_second) + aligned_third
    result[:third_start] = np.nan
    return result


def momentum(values: ArrayLike, period: int = 1) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(period)
    result = _nan_array(len(array))
    result[period:] = array[period:] - array[:-period]
    return result


def roc(values: ArrayLike, period: int = 1) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(period)
    result = _nan_array(len(array))
    base = array[:-period]
    with np.errstate(divide="ignore", invalid="ignore"):
        result[period:] = ((array[period:] - base) / base) * 100.0
    return result


def volatility(values: ArrayLike, period: int, *, ddof: int = 0) -> np.ndarray:
    array = _as_float_array(values)
    result = _nan_array(len(array))
    returns = _nan_array(len(array) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:] = np.diff(array) / array[:-1]
    result[1:] = _rolling_std(returns, period, ddof=ddof)
    return result


def true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    previous_close = np.roll(close_array, 1)
    previous_close[0] = close_array[0]
    high_low = high_array - low_array
    high_close = np.abs(high_array - previous_close)
    low_close = np.abs(low_array - previous_close)
    return np.maximum(high_low, np.maximum(high_close, low_close))


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> np.ndarray:
    return _wilder_smoothing(true_range(high, low, close), period)


def bollinger_bands(values: ArrayLike, period: int = 20, std_dev: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = _as_float_array(values)
    middle = sma(array, period)
    std = _rolling_std(array, period)
    lower = middle - (std_dev * std)
    upper = middle + (std_dev * std)
    return lower, middle, upper


def macd(values: ArrayLike, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")
    array = _as_float_array(values)
    line = ema(array, fast_period) - ema(array, slow_period)
    signal = _nan_array(len(line))
    valid_idx = np.where(~np.isnan(line))[0]
    if len(valid_idx) >= signal_period:
        signal_valid = _ema_from_array(line[valid_idx], signal_period)
        start = valid_idx[0] + signal_period - 1
        signal[start:] = signal_valid[signal_period - 1 :]
    histogram = line - signal
    return line, signal, histogram


def rsi(values: ArrayLike, period: int = 14) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(period)
    result = _nan_array(len(array))
    if len(array) <= period:
        return result
    delta = np.diff(array)
    gains = np.where(delta > 0.0, delta, 0.0)
    losses = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = _wilder_smoothing(gains, period)
    avg_loss = _wilder_smoothing(losses, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_values = 100.0 - (100.0 / (1.0 + rs))
    rsi_values = np.where(avg_loss == 0.0, 100.0, rsi_values)
    result[1:] = rsi_values
    return result


def stochastic_oscillator(high: ArrayLike, low: ArrayLike, close: ArrayLike, k_period: int = 14, d_period: int = 3) -> tuple[np.ndarray, np.ndarray]:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    lowest = _rolling_min(low_array, k_period)
    highest = _rolling_max(high_array, k_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_k = ((close_array - lowest) / (highest - lowest)) * 100.0
    percent_d = sma(percent_k, d_period)
    return percent_k, percent_d


def cci(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 20) -> np.ndarray:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    typical = (high_array + low_array + close_array) / 3.0
    mean_tp = sma(typical, period)
    result = _nan_array(len(typical))
    if len(typical) < period:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(typical, period)
    deviations = np.mean(np.abs(windows - mean_tp[period - 1 :, None]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result[period - 1 :] = (typical[period - 1 :] - mean_tp[period - 1 :]) / (0.015 * deviations)
    return result


def williams_r(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> np.ndarray:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    highest = _rolling_max(high_array, period)
    lowest = _rolling_min(low_array, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((highest - close_array) / (highest - lowest)) * -100.0


def obv(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    close_array = _as_float_array(close)
    volume_array = _as_float_array(volume)
    _validate_same_length(close_array, volume_array)
    result = np.zeros(len(close_array), dtype=np.float64)
    for i in range(1, len(close_array)):
        if close_array[i] > close_array[i - 1]:
            result[i] = result[i - 1] + volume_array[i]
        elif close_array[i] < close_array[i - 1]:
            result[i] = result[i - 1] - volume_array[i]
        else:
            result[i] = result[i - 1]
    return result


def mfi(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, period: int = 14) -> np.ndarray:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    volume_array = _as_float_array(volume)
    _validate_same_length(high_array, low_array, close_array, volume_array)
    typical = (high_array + low_array + close_array) / 3.0
    flow = typical * volume_array
    positive = np.zeros(len(typical), dtype=np.float64)
    negative = np.zeros(len(typical), dtype=np.float64)
    delta = np.diff(typical)
    positive[1:] = np.where(delta > 0.0, flow[1:], 0.0)
    negative[1:] = np.where(delta < 0.0, flow[1:], 0.0)
    positive_sum = _rolling_sum(positive, period)
    negative_sum = _rolling_sum(negative, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        money_ratio = positive_sum / negative_sum
        result = 100.0 - (100.0 / (1.0 + money_ratio))
    result = np.where(negative_sum == 0.0, 100.0, result)
    return result


def adx(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    up_move = np.diff(high_array, prepend=high_array[0])
    down_move = -np.diff(low_array, prepend=low_array[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    atr_values = atr(high_array, low_array, close_array, period)
    plus_smoothed = _wilder_smoothing(plus_dm, period)
    minus_smoothed = _wilder_smoothing(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = (plus_smoothed / atr_values) * 100.0
        minus_di = (minus_smoothed / atr_values) * 100.0
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0
    adx_values = _wilder_smoothing(np.nan_to_num(dx, nan=0.0), period)
    adx_values[np.isnan(dx)] = np.nan
    return adx_values, plus_di, minus_di


def ichimoku_cloud(high: ArrayLike, low: ArrayLike, close: ArrayLike, conversion_period: int = 9, base_period: int = 26, span_b_period: int = 52, displacement: int = 26) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    conversion = (_rolling_max(high_array, conversion_period) + _rolling_min(low_array, conversion_period)) / 2.0
    base = (_rolling_max(high_array, base_period) + _rolling_min(low_array, base_period)) / 2.0
    span_a = _nan_array(len(close_array))
    span_b = _nan_array(len(close_array))
    raw_span_a = (conversion + base) / 2.0
    raw_span_b = (_rolling_max(high_array, span_b_period) + _rolling_min(low_array, span_b_period)) / 2.0
    if displacement == 0:
        span_a[:] = raw_span_a
        span_b[:] = raw_span_b
    elif displacement < len(close_array):
        span_a[displacement:] = raw_span_a[:-displacement]
        span_b[displacement:] = raw_span_b[:-displacement]
    lagging = _nan_array(len(close_array))
    if displacement == 0:
        lagging[:] = close_array
    elif displacement < len(close_array):
        lagging[:-displacement] = close_array[displacement:]
    return conversion, base, span_a, span_b, lagging


def keltner_channels(high: ArrayLike, low: ArrayLike, close: ArrayLike, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    middle = ema(close, ema_period)
    atr_values = atr(high, low, close, atr_period)
    lower = middle - (multiplier * atr_values)
    upper = middle + (multiplier * atr_values)
    return lower, middle, upper


def donchian_channels(high: ArrayLike, low: ArrayLike, period: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    _validate_same_length(high_array, low_array)
    upper = _rolling_max(high_array, period)
    lower = _rolling_min(low_array, period)
    middle = (lower + upper) / 2.0
    return lower, middle, upper


def aroon(high: ArrayLike, low: ArrayLike, period: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    _validate_same_length(high_array, low_array)
    _validate_period(period)
    up = _nan_array(len(high_array))
    down = _nan_array(len(high_array))
    oscillator = _nan_array(len(high_array))
    if len(high_array) < period:
        return up, down, oscillator
    high_windows = np.lib.stride_tricks.sliding_window_view(high_array, period)
    low_windows = np.lib.stride_tricks.sliding_window_view(low_array, period)
    since_high = period - 1 - np.argmax(high_windows, axis=1)
    since_low = period - 1 - np.argmin(low_windows, axis=1)
    up[period - 1 :] = ((period - since_high) / period) * 100.0
    down[period - 1 :] = ((period - since_low) / period) * 100.0
    oscillator[period - 1 :] = up[period - 1 :] - down[period - 1 :]
    return up, down, oscillator


def vortex(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> tuple[np.ndarray, np.ndarray]:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    prev_low = np.roll(low_array, 1)
    prev_high = np.roll(high_array, 1)
    prev_low[0] = low_array[0]
    prev_high[0] = high_array[0]
    vm_plus = np.abs(high_array - prev_low)
    vm_minus = np.abs(low_array - prev_high)
    tr_sum = _rolling_sum(true_range(high_array, low_array, close_array), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        vi_plus = _rolling_sum(vm_plus, period) / tr_sum
        vi_minus = _rolling_sum(vm_minus, period) / tr_sum
    return vi_plus, vi_minus


def ultimate_oscillator(high: ArrayLike, low: ArrayLike, close: ArrayLike, short_period: int = 7, medium_period: int = 14, long_period: int = 28) -> np.ndarray:
    if not (short_period < medium_period < long_period):
        raise ValueError("periods must satisfy short < medium < long")
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    _validate_same_length(high_array, low_array, close_array)
    prev_close = np.roll(close_array, 1)
    prev_close[0] = close_array[0]
    buying_pressure = close_array - np.minimum(low_array, prev_close)
    true_low = np.minimum(low_array, prev_close)
    true_high = np.maximum(high_array, prev_close)
    true_range_values = true_high - true_low
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_short = _rolling_sum(buying_pressure, short_period) / _rolling_sum(true_range_values, short_period)
        avg_medium = _rolling_sum(buying_pressure, medium_period) / _rolling_sum(true_range_values, medium_period)
        avg_long = _rolling_sum(buying_pressure, long_period) / _rolling_sum(true_range_values, long_period)
    return 100.0 * ((4.0 * avg_short) + (2.0 * avg_medium) + avg_long) / 7.0


def trix(values: ArrayLike, period: int = 15) -> np.ndarray:
    array = _as_float_array(values)
    result = _nan_array(len(array))
    ema1 = ema(array, period)
    ema1_idx = np.where(~np.isnan(ema1))[0]
    if len(ema1_idx) < period:
        return result
    ema2 = _ema_from_array(ema1[ema1_idx], period)
    ema2_idx = np.where(~np.isnan(ema2))[0]
    if len(ema2_idx) < period:
        return result
    ema3 = _ema_from_array(ema2[ema2_idx], period)
    ema3_idx = np.where(~np.isnan(ema3))[0]
    if len(ema3_idx) == 0:
        return result
    start = ema1_idx[0] + ema2_idx[0] + ema3_idx[0]
    aligned = _nan_array(len(array))
    aligned[start:] = ema3[ema3_idx[0] :]
    with np.errstate(divide="ignore", invalid="ignore"):
        result[1:] = ((aligned[1:] - aligned[:-1]) / aligned[:-1]) * 100.0
    return result


def kama(values: ArrayLike, er_period: int = 10, fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(er_period)
    _validate_period(fast_period)
    _validate_period(slow_period)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")
    result = _nan_array(len(array))
    if len(array) <= er_period:
        return result
    fastest = 2.0 / (fast_period + 1.0)
    slowest = 2.0 / (slow_period + 1.0)
    result[er_period] = array[er_period]
    for i in range(er_period + 1, len(array)):
        change = abs(array[i] - array[i - er_period])
        volatility_sum = np.sum(np.abs(np.diff(array[i - er_period : i + 1])))
        efficiency_ratio = 0.0 if volatility_sum == 0.0 else change / volatility_sum
        smoothing = (efficiency_ratio * (fastest - slowest) + slowest) ** 2
        result[i] = result[i - 1] + smoothing * (array[i] - result[i - 1])
    return result


def zscore(values: ArrayLike, period: int = 20) -> np.ndarray:
    array = _as_float_array(values)
    mean = sma(array, period)
    std = _rolling_std(array, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (array - mean) / std


def sharpe_ratio(values: ArrayLike, period: int = 20, risk_free_rate: float = 0.0) -> np.ndarray:
    array = _as_float_array(values)
    returns = _nan_array(len(array) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:] = np.diff(array) / array[:-1]
    excess = returns - risk_free_rate
    mean = _rolling_mean(excess, period)
    std = _rolling_std(excess, period, ddof=1)
    result = _nan_array(len(array))
    with np.errstate(divide="ignore", invalid="ignore"):
        result[1:] = mean / std
    return result


def sortino_ratio(values: ArrayLike, period: int = 20, target_return: float = 0.0) -> np.ndarray:
    array = _as_float_array(values)
    returns = _nan_array(len(array) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:] = np.diff(array) / array[:-1]
    excess = returns - target_return
    mean = _rolling_mean(excess, period)
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(_rolling_mean(downside * downside, period))
    result = _nan_array(len(array))
    with np.errstate(divide="ignore", invalid="ignore"):
        result[1:] = mean / downside_std
    return result


def hurst_exponent(values: ArrayLike, period: int = 100) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(period, minimum=20)
    result = _nan_array(len(array))
    if len(array) < period:
        return result
    for end in range(period - 1, len(array)):
        window = array[end - period + 1 : end + 1]
        centered = window - np.mean(window)
        cumulative = np.cumsum(centered)
        spread = np.max(cumulative) - np.min(cumulative)
        std = np.std(window)
        if spread == 0.0 or std == 0.0:
            continue
        result[end] = np.log(spread / std) / np.log(period)
    return result


def fisher_transform(values: ArrayLike, period: int = 10) -> np.ndarray:
    array = _as_float_array(values)
    highest = _rolling_max(array, period)
    lowest = _rolling_min(array, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = 2.0 * ((array - lowest) / (highest - lowest) - 0.5)
    clipped = np.clip(np.nan_to_num(normalized, nan=0.0), -0.999, 0.999)
    result = _nan_array(len(array))
    result[period - 1 :] = 0.5 * np.log((1.0 + clipped[period - 1 :]) / (1.0 - clipped[period - 1 :]))
    return result


def linear_regression_slope(values: ArrayLike, period: int = 20) -> np.ndarray:
    array = _as_float_array(values)
    _validate_period(period)
    result = _nan_array(len(array))
    if len(array) < period:
        return result
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    windows = np.lib.stride_tricks.sliding_window_view(array, period)
    y_mean = np.mean(windows, axis=1)
    numerator = np.sum((x - x_mean) * (windows - y_mean[:, None]), axis=1)
    result[period - 1 :] = numerator / denominator
    return result


class IndicatorCatalog:
    def __init__(self):
        self.sma = sma
        self.ema = ema
        self.wma = wma
        self.dema = dema
        self.tema = tema
        self.momentum = momentum
        self.roc = roc
        self.volatility = volatility
        self.true_range = true_range
        self.atr = atr
        self.bollinger_bands = bollinger_bands
        self.macd = macd
        self.rsi = rsi
        self.stochastic_oscillator = stochastic_oscillator
        self.cci = cci
        self.williams_r = williams_r
        self.obv = obv
        self.mfi = mfi
        self.adx = adx
        self.ichimoku_cloud = ichimoku_cloud
        self.keltner_channels = keltner_channels
        self.donchian_channels = donchian_channels
        self.aroon = aroon
        self.vortex = vortex
        self.ultimate_oscillator = ultimate_oscillator
        self.trix = trix
        self.kama = kama
        self.zscore = zscore
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio
        self.hurst_exponent = hurst_exponent
        self.fisher_transform = fisher_transform
        self.linear_regression_slope = linear_regression_slope


indicators = IndicatorCatalog()
