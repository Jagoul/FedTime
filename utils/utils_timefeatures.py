from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0: 
    > * m - [month]
    > * w - [month]
    > * d - [month, day]
    > * b - [month, day]
    > * h - [month, day, hour]
    > * t - [month, day, hour, *minute]
    > 
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]): 
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if freq == 'y':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('A')]).transpose(1, 0)
    elif freq == 'm':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('M')]).transpose(1, 0)
    elif freq == 'w':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('W')]).transpose(1, 0)
    elif freq == 'd':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('D')]).transpose(1, 0)
    elif freq == 'b':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('B')]).transpose(1, 0)
    elif freq == 'h':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('H')]).transpose(1, 0)
    elif freq == 't':
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('T')]).transpose(1, 0)
    else:
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str('S')]).transpose(1, 0)


# Additional utility functions for time features
def get_sincos_encoding(values, max_val):
    """
    Get sine-cosine encoding for cyclical features
    
    Args:
        values: Array of values to encode
        max_val: Maximum value in the cycle
    
    Returns:
        Sine and cosine encodings
    """
    normalized = 2 * np.pi * values / max_val
    return np.sin(normalized), np.cos(normalized)


def encode_cyclical_feature(feature_values, period):
    """
    Encode a cyclical feature using sine and cosine
    
    Args:
        feature_values: Values to encode (e.g., hour of day)
        period: Period of the cycle (e.g., 24 for hour of day)
    
    Returns:
        Sine and cosine encodings as a 2D array
    """
    sin_vals, cos_vals = get_sincos_encoding(feature_values, period)
    return np.column_stack([sin_vals, cos_vals])


def create_time_features_dataframe(timestamps, freq='H'):
    """
    Create a DataFrame with time features from timestamps
    
    Args:
        timestamps: Pandas DatetimeIndex or Series
        freq: Frequency string
    
    Returns:
        DataFrame with time features
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.to_datetime(timestamps)
    
    features = time_features(timestamps, freq.lower())
    feature_names = [feat.__class__.__name__ for feat in time_features_from_frequency_str(freq)]
    
    return pd.DataFrame(features, columns=feature_names, index=timestamps)


def add_lag_features(df, target_col, lags):
    """
    Add lag features to a DataFrame
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        lags: List of lag values
    
    Returns:
        DataFrame with lag features added
    """
    df_with_lags = df.copy()
    
    for lag in lags:
        df_with_lags[f'{target_col}_lag_{lag}'] = df_with_lags[target_col].shift(lag)
    
    return df_with_lags


def add_rolling_features(df, target_col, windows, operations=['mean', 'std']):
    """
    Add rolling window features to a DataFrame
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        windows: List of window sizes
        operations: List of operations to apply
    
    Returns:
        DataFrame with rolling features added
    """
    df_with_rolling = df.copy()
    
    for window in windows:
        for op in operations:
            if op == 'mean':
                df_with_rolling[f'{target_col}_rolling_{window}_mean'] = df_with_rolling[target_col].rolling(window).mean()
            elif op == 'std':
                df_with_rolling[f'{target_col}_rolling_{window}_std'] = df_with_rolling[target_col].rolling(window).std()
            elif op == 'min':
                df_with_rolling[f'{target_col}_rolling_{window}_min'] = df_with_rolling[target_col].rolling(window).min()
            elif op == 'max':
                df_with_rolling[f'{target_col}_rolling_{window}_max'] = df_with_rolling[target_col].rolling(window).max()
    
    return df_with_rolling


def create_holiday_features(timestamps, country='US'):
    """
    Create holiday features for given timestamps
    
    Args:
        timestamps: Pandas DatetimeIndex
        country: Country code for holidays
    
    Returns:
        Binary array indicating holidays
    """
    try:
        import holidays
        country_holidays = holidays.country_holidays(country)
        is_holiday = timestamps.date.isin(country_holidays)
        return is_holiday.astype(int)
    except ImportError:
        print("holidays package not installed, returning zeros")
        return np.zeros(len(timestamps))


def get_business_day_features(timestamps):
    """
    Get business day features
    
    Args:
        timestamps: Pandas DatetimeIndex
    
    Returns:
        Dictionary with business day features
    """
    features = {}
    features['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
    features['is_business_day'] = (timestamps.dayofweek < 5).astype(int)
    features['days_since_weekend'] = np.where(
        timestamps.dayofweek < 5,
        timestamps.dayofweek,
        0
    )
    features['days_until_weekend'] = np.where(
        timestamps.dayofweek < 5,
        4 - timestamps.dayofweek,
        0
    )
    
    return features