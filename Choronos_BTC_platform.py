
import os
import time
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


# ============================================================
# 1. STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chronos-2 BTC/USD Signal App",
    page_icon="₿",
    layout="wide",
)

st.title("₿ Chronos-2 BTC/USD Forecast Signal App")
st.caption(
    "Uses your trained AutoGluon/Chronos-2 model, recent Bitstamp data, "
    "and backtest performance to generate BUY / SELL / HOLD signals."
)

st.warning(
    "Research / paper-trading tool only. This app does not place trades and is not financial advice."
)


# ============================================================
# 2. DEFAULT CONFIG
# ============================================================

DEFAULT_GIT_REPO_DIR = "bitstamp-btcusd-minute-data"

DEFAULT_HISTORICAL_CSV = (
    "bitstamp-btcusd-minute-data/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
)

DEFAULT_RECENT_CSV = (
    "bitstamp-btcusd-minute-data/data/updates/btcusd_bitstamp_1min_latest.csv"
)

DEFAULT_MODEL_PATH = "models/btc_chronos2_bitstamp"

DEFAULT_BACKTEST_CSV = (
    "chronos2_bitstamp_training_outputs/chronos2_bitstamp_backtest_predictions.csv"
)

DEFAULT_SIGNAL_LOG_PATH = "chronos2_streamlit_signal_log.csv"

DEFAULT_SERIES_ID = "BTC-USD"

DEFAULT_COVARIATE_COLS = ["volume", "RSI", "MACD", "ATR", "ADX"]


# ============================================================
# 3. GENERAL UTILITIES
# ============================================================

def utc_now_naive():
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def get_file_mtime(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_mtime


def clean_col_name(name):
    return "".join(ch for ch in str(name).lower().strip() if ch.isalnum())


def find_column(columns, candidates, required=True):
    normalized = {clean_col_name(c): c for c in columns}

    for cand in candidates:
        key = clean_col_name(cand)
        if key in normalized:
            return normalized[key]

    for col in columns:
        col_key = clean_col_name(col)
        for cand in candidates:
            cand_key = clean_col_name(cand)
            if cand_key and cand_key in col_key:
                return col

    if required:
        raise ValueError(
            f"Could not find required column. Candidates={candidates}, "
            f"available columns={list(columns)}"
        )

    return None


def parse_timestamp_column(series):
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio > 0.90:
        median_value = numeric.dropna().median()

        if median_value > 1e12:
            ts = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(series, utc=True, errors="coerce")

    return ts.dt.tz_convert(None)


def maybe_git_pull(git_repo_dir):
    git_repo_dir = Path(git_repo_dir)

    if not git_repo_dir.exists():
        st.warning(f"Git repo folder not found: {git_repo_dir}")
        return

    try:
        result = subprocess.run(
            ["git", "-C", str(git_repo_dir), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.stdout.strip():
            st.info(result.stdout)

        if result.stderr.strip():
            st.warning(result.stderr)

    except Exception as e:
        st.warning(f"git pull failed, continuing with current files. Error: {e}")


# ============================================================
# 4. DATA LOADING AND STANDARDIZATION
# ============================================================

def standardize_bitstamp_dataframe(raw_df, series_id):
    """
    Converts possible Bitstamp schemas into:

        id, timestamp, open, high, low, target, volume

    target = close price
    """

    timestamp_col = find_column(
        raw_df.columns,
        ["timestamp", "unix", "unix_timestamp", "date", "datetime", "time"],
        required=True,
    )

    open_col = find_column(raw_df.columns, ["open"], required=False)
    high_col = find_column(raw_df.columns, ["high"], required=False)
    low_col = find_column(raw_df.columns, ["low"], required=False)

    close_col = find_column(
        raw_df.columns,
        ["close", "target", "weighted_price", "weightedprice", "price"],
        required=True,
    )

    volume_col = find_column(
        raw_df.columns,
        [
            "volume_btc",
            "volume btc",
            "volume_(btc)",
            "volumebtc",
            "base_volume",
            "basevolume",
            "amount",
            "volume",
        ],
        required=False,
    )

    out = pd.DataFrame()
    out["timestamp"] = parse_timestamp_column(raw_df[timestamp_col])
    out["target"] = pd.to_numeric(raw_df[close_col], errors="coerce")

    if open_col is not None:
        out["open"] = pd.to_numeric(raw_df[open_col], errors="coerce")
    else:
        out["open"] = out["target"]

    if high_col is not None:
        out["high"] = pd.to_numeric(raw_df[high_col], errors="coerce")
    else:
        out["high"] = out["target"]

    if low_col is not None:
        out["low"] = pd.to_numeric(raw_df[low_col], errors="coerce")
    else:
        out["low"] = out["target"]

    if volume_col is not None:
        out["volume"] = pd.to_numeric(raw_df[volume_col], errors="coerce")
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["timestamp", "target"])
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.reset_index(drop=True)

    out["id"] = series_id

    return out[["id", "timestamp", "open", "high", "low", "target", "volume"]]


@st.cache_data(show_spinner=False)
def load_bitstamp_dataset_cached(
    historical_csv,
    recent_csv,
    historical_mtime,
    recent_mtime,
    series_id,
    prefer_recent_only,
    minimum_recent_rows,
):
    """
    Loads recent CSV only if it has enough rows for context.
    Otherwise loads historical + recent.

    This is useful because the full dataset has 7M+ rows.
    """

    historical_csv = Path(historical_csv)
    recent_csv = Path(recent_csv)

    if not recent_csv.exists():
        raise FileNotFoundError(f"Recent CSV not found: {recent_csv}")

    raw_recent = pd.read_csv(recent_csv, low_memory=False)
    df_recent = standardize_bitstamp_dataframe(raw_recent, series_id)
    del raw_recent

    if prefer_recent_only and len(df_recent) >= minimum_recent_rows:
        df_full = df_recent.copy()
        source_used = "recent_only"
    else:
        if not historical_csv.exists():
            raise FileNotFoundError(f"Historical CSV not found: {historical_csv}")

        raw_hist = pd.read_csv(historical_csv, compression="infer", low_memory=False)
        df_hist = standardize_bitstamp_dataframe(raw_hist, series_id)
        del raw_hist

        df_full = pd.concat([df_hist, df_recent], ignore_index=True)
        source_used = "historical_plus_recent"

    df_full = df_full.sort_values("timestamp")
    df_full = df_full.drop_duplicates(subset=["timestamp"], keep="last")
    df_full = df_full.reset_index(drop=True)

    return df_full, source_used


# ============================================================
# 5. TECHNICAL INDICATORS
# ============================================================

def compute_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    return ema_fast - ema_slow


def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0,
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0,
    )

    atr = compute_atr(high, low, close, period)

    plus_di = (
        100
        * pd.Series(plus_dm, index=high.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * pd.Series(minus_dm, index=high.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr.replace(0, np.nan)
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute_indicators(df, covariate_cols):
    df = df.copy()

    if "RSI" in covariate_cols:
        df["RSI"] = compute_rsi(df["target"])

    if "MACD" in covariate_cols:
        df["MACD"] = compute_macd(df["target"])

    if "ATR" in covariate_cols:
        df["ATR"] = compute_atr(df["high"], df["low"], df["target"])

    if "ADX" in covariate_cols:
        df["ADX"] = compute_adx(df["high"], df["low"], df["target"])

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def make_regular_1min(df, series_id):
    df = df.copy()
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last")
    df = df.set_index("timestamp")

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1min",
    )

    df = df.reindex(full_index)
    df.index.name = "timestamp"

    for col in ["open", "high", "low", "target"]:
        df[col] = df[col].ffill()

    df["volume"] = df["volume"].fillna(0.0)
    df["id"] = series_id

    df = df.dropna(subset=["target"])
    df = df.reset_index()

    return df[["id", "timestamp", "open", "high", "low", "target", "volume"]]


def resample_ohlcv(df, rule, series_id):
    df = df.copy()
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    out = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "target": "last",
            "volume": "sum",
        }
    )

    out = out.dropna(subset=["target"]).reset_index()
    out["id"] = series_id

    return out[["id", "timestamp", "open", "high", "low", "target", "volume"]]


def prepare_context_dataframe(
    df_full,
    series_id,
    context_window,
    indicator_warmup_rows,
    covariate_cols,
    resample_rule=None,
    regularize_1min=True,
):
    needed_rows = context_window + indicator_warmup_rows

    df = df_full.sort_values("timestamp").tail(needed_rows).copy()

    if resample_rule:
        df = resample_ohlcv(df, resample_rule, series_id)
    elif regularize_1min:
        df = make_regular_1min(df, series_id)

    df = compute_indicators(df, covariate_cols)

    required_cols = ["id", "timestamp", "target"] + covariate_cols

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns after indicators: {missing}")

    df = df[required_cols].dropna().reset_index(drop=True)
    df = df.tail(context_window).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Prepared context is empty after indicator computation.")

    return df


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_predictor_cached(model_path):
    return TimeSeriesPredictor.load(model_path)


def get_model_names_safe(predictor):
    try:
        return predictor.model_names()
    except Exception:
        return []


def choose_model_name(predictor, preferred_model_name):
    if not preferred_model_name:
        return None

    model_names = get_model_names_safe(predictor)

    if preferred_model_name in model_names:
        return preferred_model_name

    return None


# ============================================================
# 7. FORECASTING
# ============================================================

def select_prediction_column(forecast_df):
    preferred = ["mean", "0.5", 0.5, "median"]

    for col in preferred:
        if col in forecast_df.columns:
            return col

    ignore = {"item_id", "id", "timestamp"}

    numeric_cols = [
        c
        for c in forecast_df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(forecast_df[c])
    ]

    if not numeric_cols:
        raise ValueError(f"No numeric prediction columns found: {list(forecast_df.columns)}")

    return numeric_cols[0]


def get_quantile_columns(forecast_df):
    qcols = []

    for c in forecast_df.columns:
        try:
            q = float(c)
        except Exception:
            continue

        if 0.0 < q < 1.0:
            qcols.append((q, c))

    qcols = sorted(qcols, key=lambda x: x[0])

    return qcols


def predict_next_minutes(
    predictor,
    model_name,
    context_df,
    series_id,
    covariate_cols,
):
    input_cols = ["id", "timestamp", "target"] + covariate_cols

    ts_context = TimeSeriesDataFrame.from_data_frame(
        context_df[input_cols],
        id_column="id",
        timestamp_column="timestamp",
    )

    if model_name is None:
        forecast = predictor.predict(ts_context)
    else:
        forecast = predictor.predict(ts_context, model=model_name)

    forecast_df = forecast.reset_index()

    if "item_id" in forecast_df.columns:
        forecast_df = forecast_df[forecast_df["item_id"] == series_id].copy()
    elif "id" in forecast_df.columns:
        forecast_df = forecast_df[forecast_df["id"] == series_id].copy()

    if "timestamp" not in forecast_df.columns:
        raise ValueError(f"Forecast output has no timestamp column: {forecast_df.columns}")

    forecast_df = forecast_df.sort_values("timestamp").reset_index(drop=True)

    pred_col = select_prediction_column(forecast_df)
    qcols = get_quantile_columns(forecast_df)

    last_known_timestamp = context_df["timestamp"].iloc[-1]
    last_known_price = float(context_df["target"].iloc[-1])

    forecast_df["step_ahead"] = np.arange(1, len(forecast_df) + 1)
    forecast_df["target_timestamp"] = forecast_df["timestamp"]
    forecast_df["last_known_timestamp"] = last_known_timestamp
    forecast_df["last_known_price"] = last_known_price
    forecast_df["predicted_price"] = forecast_df[pred_col].astype(float)
    forecast_df["predicted_change"] = forecast_df["predicted_price"] - last_known_price
    forecast_df["predicted_return_pct"] = (
        100.0 * forecast_df["predicted_change"] / last_known_price
    )

    forecast_df["predicted_direction"] = np.where(
        forecast_df["predicted_change"] > 0,
        "UP",
        np.where(forecast_df["predicted_change"] < 0, "DOWN", "FLAT"),
    )

    forecast_df["step_ahead_minutes"] = (
        pd.to_datetime(forecast_df["target_timestamp"])
        - pd.to_datetime(last_known_timestamp)
    ) / pd.Timedelta(minutes=1)

    return forecast_df, pred_col, qcols


# ============================================================
# 8. BACKTEST PERFORMANCE AND THRESHOLD CALIBRATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_backtest_csv_cached(backtest_csv, backtest_mtime):
    backtest_csv = Path(backtest_csv)

    if not backtest_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(backtest_csv)

    for col in ["timestamp", "target_timestamp", "last_known_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def prepare_backtest_for_calibration(backtest_df, horizon_step):
    if backtest_df.empty:
        return pd.DataFrame()

    df = backtest_df.copy()

    if "target_timestamp" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "target_timestamp"})

    required = ["actual", "predicted", "last_known_price"]

    for c in required:
        if c not in df.columns:
            return pd.DataFrame()

    if "step_ahead" in df.columns:
        df_h = df[df["step_ahead"] == horizon_step].copy()

        # If too few samples at selected horizon, use all rows as fallback
        if len(df_h) >= 5:
            df = df_h

    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df["last_known_price"] = pd.to_numeric(df["last_known_price"], errors="coerce")

    df = df.dropna(subset=["actual", "predicted", "last_known_price"])

    df = df[df["last_known_price"] > 0].copy()

    if len(df) == 0:
        return pd.DataFrame()

    df["pred_return_pct"] = (
        100.0 * (df["predicted"] - df["last_known_price"]) / df["last_known_price"]
    )

    df["actual_return_pct"] = (
        100.0 * (df["actual"] - df["last_known_price"]) / df["last_known_price"]
    )

    df["error"] = df["predicted"] - df["actual"]

    df["error_pct"] = 100.0 * df["error"] / df["last_known_price"]

    df["absolute_error_pct"] = df["error_pct"].abs()

    df["direction_correct"] = (
        np.sign(df["pred_return_pct"]) == np.sign(df["actual_return_pct"])
    )

    return df.reset_index(drop=True)


def side_precision_at_threshold(df, side, threshold_pct, transaction_cost_pct):
    if df.empty:
        return {
            "support": 0,
            "precision": np.nan,
        }

    if side == "BUY":
        signal_mask = df["pred_return_pct"] >= threshold_pct
        correct_mask = df["actual_return_pct"] > transaction_cost_pct

    elif side == "SELL":
        signal_mask = df["pred_return_pct"] <= -threshold_pct
        correct_mask = df["actual_return_pct"] < -transaction_cost_pct

    else:
        raise ValueError("side must be BUY or SELL")

    support = int(signal_mask.sum())

    if support == 0:
        precision = np.nan
    else:
        precision = float(correct_mask[signal_mask].mean())

    return {
        "support": support,
        "precision": precision,
    }


def calibrate_side_threshold(
    df,
    side,
    fallback_threshold_pct,
    min_threshold_pct,
    min_precision,
    min_samples,
    transaction_cost_pct,
):
    if df.empty:
        return fallback_threshold_pct, {
            "support": 0,
            "precision": np.nan,
            "source": "fallback_empty_backtest",
        }

    abs_pred = df["pred_return_pct"].abs().dropna()

    if len(abs_pred) == 0:
        return fallback_threshold_pct, {
            "support": 0,
            "precision": np.nan,
            "source": "fallback_no_predictions",
        }

    max_candidate = max(float(abs_pred.quantile(0.95)), fallback_threshold_pct)

    grid = np.linspace(min_threshold_pct, max_candidate, 100)

    quantile_grid = abs_pred.quantile(np.linspace(0.05, 0.95, 50)).values

    candidates = np.unique(
        np.concatenate(
            [
                grid,
                quantile_grid,
                np.array([fallback_threshold_pct, min_threshold_pct]),
            ]
        )
    )

    candidates = candidates[np.isfinite(candidates)]
    candidates = candidates[candidates >= min_threshold_pct]
    candidates = np.sort(candidates)

    chosen_threshold = None
    chosen_stats = None

    for t in candidates:
        stats = side_precision_at_threshold(
            df=df,
            side=side,
            threshold_pct=float(t),
            transaction_cost_pct=transaction_cost_pct,
        )

        if stats["support"] >= min_samples and np.isfinite(stats["precision"]):
            if stats["precision"] >= min_precision:
                chosen_threshold = float(t)
                chosen_stats = stats
                break

    if chosen_threshold is None:
        chosen_threshold = float(fallback_threshold_pct)
        chosen_stats = side_precision_at_threshold(
            df=df,
            side=side,
            threshold_pct=chosen_threshold,
            transaction_cost_pct=transaction_cost_pct,
        )
        chosen_stats["source"] = "fallback_no_threshold_met_precision"
    else:
        chosen_stats["source"] = "calibrated_from_backtest"

    return chosen_threshold, chosen_stats


def build_calibration(
    backtest_df,
    horizon_step,
    fallback_threshold_pct,
    min_threshold_pct,
    mae_multiplier,
    volatility_multiplier,
    target_direction_accuracy,
    min_precision,
    min_samples,
    transaction_cost_pct,
):
    df = prepare_backtest_for_calibration(backtest_df, horizon_step)

    if df.empty:
        return {
            "usable": False,
            "calibration_df": df,
            "rows": 0,
            "mae_pct": np.nan,
            "rmse_pct": np.nan,
            "trend_accuracy": np.nan,
            "median_abs_actual_move_pct": np.nan,
            "base_dynamic_threshold_pct": fallback_threshold_pct,
            "buy_threshold_pct": fallback_threshold_pct,
            "sell_threshold_pct": fallback_threshold_pct,
            "buy_stats": {"support": 0, "precision": np.nan, "source": "fallback"},
            "sell_stats": {"support": 0, "precision": np.nan, "source": "fallback"},
        }

    mae_pct = float(df["absolute_error_pct"].mean())
    rmse_pct = float(np.sqrt(np.mean(np.square(df["error_pct"]))))
    trend_accuracy = float(df["direction_correct"].mean())
    median_abs_actual_move_pct = float(df["actual_return_pct"].abs().median())

    base_threshold = max(
        min_threshold_pct,
        fallback_threshold_pct,
        mae_multiplier * mae_pct,
        volatility_multiplier * median_abs_actual_move_pct,
    )

    if np.isfinite(trend_accuracy) and trend_accuracy > 0:
        performance_factor = target_direction_accuracy / trend_accuracy
        performance_factor = float(np.clip(performance_factor, 0.75, 2.50))
    else:
        performance_factor = 1.50

    base_dynamic_threshold = base_threshold * performance_factor

    buy_threshold, buy_stats = calibrate_side_threshold(
        df=df,
        side="BUY",
        fallback_threshold_pct=base_dynamic_threshold,
        min_threshold_pct=min_threshold_pct,
        min_precision=min_precision,
        min_samples=min_samples,
        transaction_cost_pct=transaction_cost_pct,
    )

    sell_threshold, sell_stats = calibrate_side_threshold(
        df=df,
        side="SELL",
        fallback_threshold_pct=base_dynamic_threshold,
        min_threshold_pct=min_threshold_pct,
        min_precision=min_precision,
        min_samples=min_samples,
        transaction_cost_pct=transaction_cost_pct,
    )

    return {
        "usable": True,
        "calibration_df": df,
        "rows": len(df),
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "trend_accuracy": trend_accuracy,
        "median_abs_actual_move_pct": median_abs_actual_move_pct,
        "base_dynamic_threshold_pct": float(base_dynamic_threshold),
        "buy_threshold_pct": float(buy_threshold),
        "sell_threshold_pct": float(sell_threshold),
        "buy_stats": buy_stats,
        "sell_stats": sell_stats,
    }


# ============================================================
# 9. CONFIDENCE AND SIGNAL LOGIC
# ============================================================

def quantile_consensus(row, qcols, last_known_price, side):
    if not qcols:
        return 0.50

    values = []

    for _, col in qcols:
        try:
            v = float(row[col])
            if np.isfinite(v):
                values.append(v)
        except Exception:
            pass

    if len(values) == 0:
        return 0.50

    values = np.array(values)

    if side == "BUY":
        return float(np.mean(values > last_known_price))

    if side == "SELL":
        return float(np.mean(values < last_known_price))

    return 0.50


def estimate_backtest_precision_for_current_signal(
    calibration,
    side,
    current_abs_pred_return_pct,
    side_threshold_pct,
    min_samples,
    transaction_cost_pct,
):
    df = calibration.get("calibration_df", pd.DataFrame())

    if df.empty:
        trend_acc = calibration.get("trend_accuracy", np.nan)
        if np.isfinite(trend_acc):
            return float(trend_acc), 0
        return 0.50, 0

    if side == "BUY":
        side_mask = df["pred_return_pct"] > 0
        correct_mask = df["actual_return_pct"] > transaction_cost_pct
    elif side == "SELL":
        side_mask = df["pred_return_pct"] < 0
        correct_mask = df["actual_return_pct"] < -transaction_cost_pct
    else:
        return 0.50, 0

    candidate_thresholds = [
        max(side_threshold_pct, 0.75 * current_abs_pred_return_pct),
        side_threshold_pct,
        0.0,
    ]

    for t in candidate_thresholds:
        if side == "BUY":
            mask = side_mask & (df["pred_return_pct"] >= t)
        else:
            mask = side_mask & (df["pred_return_pct"] <= -t)

        support = int(mask.sum())

        if support >= min_samples:
            precision = float(correct_mask[mask].mean())
            return precision, support

    mask = side_mask
    support = int(mask.sum())

    if support > 0:
        precision = float(correct_mask[mask].mean())
        return precision, support

    return 0.50, 0


def generate_signal(
    forecast_df,
    horizon_step,
    qcols,
    calibration,
    min_signal_confidence,
    min_samples_for_confidence,
    transaction_cost_pct,
    confidence_threshold_min_factor,
    confidence_threshold_max_factor,
):
    selected = forecast_df[forecast_df["step_ahead"] == horizon_step].copy()

    if len(selected) == 0:
        selected = forecast_df.tail(1).copy()

    row = selected.iloc[0]

    last_known_price = float(row["last_known_price"])
    pred_price = float(row["predicted_price"])
    pred_return_pct = float(row["predicted_return_pct"])
    abs_pred_return_pct = abs(pred_return_pct)

    if pred_return_pct > 0:
        side = "BUY"
        base_threshold_pct = calibration["buy_threshold_pct"]
    elif pred_return_pct < 0:
        side = "SELL"
        base_threshold_pct = calibration["sell_threshold_pct"]
    else:
        side = "HOLD"
        base_threshold_pct = calibration["base_dynamic_threshold_pct"]

    if side == "HOLD":
        return {
            "signal": "HOLD",
            "raw_side": "FLAT",
            "reason": "Predicted move is exactly flat.",
            "confidence": 0.50,
            "confidence_pct": 50.0,
            "base_threshold_pct": float(base_threshold_pct),
            "effective_threshold_pct": float(base_threshold_pct),
            "predicted_return_pct": pred_return_pct,
            "predicted_price": pred_price,
            "last_known_price": last_known_price,
            "target_timestamp": row["target_timestamp"],
            "horizon_step": int(row["step_ahead"]),
            "backtest_precision": np.nan,
            "backtest_support": 0,
            "quantile_consensus": np.nan,
            "magnitude_score": 0.0,
        }

    backtest_precision, support = estimate_backtest_precision_for_current_signal(
        calibration=calibration,
        side=side,
        current_abs_pred_return_pct=abs_pred_return_pct,
        side_threshold_pct=base_threshold_pct,
        min_samples=min_samples_for_confidence,
        transaction_cost_pct=transaction_cost_pct,
    )

    support_weight = min(1.0, support / max(min_samples_for_confidence, 1))

    backtest_precision_shrunk = (
        support_weight * backtest_precision + (1.0 - support_weight) * 0.50
    )

    q_consensus = quantile_consensus(
        row=row,
        qcols=qcols,
        last_known_price=last_known_price,
        side=side,
    )

    magnitude_score = min(
        1.0,
        abs_pred_return_pct / max(base_threshold_pct, 1e-9),
    )

    confidence = (
        0.60 * backtest_precision_shrunk
        + 0.25 * q_consensus
        + 0.15 * magnitude_score
    )

    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Dynamic threshold:
    # - if confidence is high, threshold decreases
    # - if confidence is low, threshold increases
    confidence_factor = min_signal_confidence / max(confidence, 1e-6)

    confidence_factor = float(
        np.clip(
            confidence_factor,
            confidence_threshold_min_factor,
            confidence_threshold_max_factor,
        )
    )

    effective_threshold_pct = float(base_threshold_pct * confidence_factor)

    if side == "BUY":
        passed_move = pred_return_pct >= effective_threshold_pct
    else:
        passed_move = pred_return_pct <= -effective_threshold_pct

    passed_confidence = confidence >= min_signal_confidence

    if passed_move and passed_confidence:
        signal = side
        reason = (
            f"{side} because predicted move {pred_return_pct:.5f}% passed "
            f"effective threshold {effective_threshold_pct:.5f}% and confidence "
            f"{confidence:.2%} passed minimum {min_signal_confidence:.2%}."
        )
    else:
        signal = "HOLD"

        reasons = []

        if not passed_move:
            reasons.append(
                f"predicted move {pred_return_pct:.5f}% did not pass "
                f"effective threshold ±{effective_threshold_pct:.5f}%"
            )

        if not passed_confidence:
            reasons.append(
                f"confidence {confidence:.2%} is below minimum "
                f"{min_signal_confidence:.2%}"
            )

        reason = "HOLD because " + " and ".join(reasons) + "."

    return {
        "signal": signal,
        "raw_side": side,
        "reason": reason,
        "confidence": confidence,
        "confidence_pct": 100.0 * confidence,
        "base_threshold_pct": float(base_threshold_pct),
        "effective_threshold_pct": float(effective_threshold_pct),
        "predicted_return_pct": pred_return_pct,
        "predicted_price": pred_price,
        "last_known_price": last_known_price,
        "target_timestamp": row["target_timestamp"],
        "horizon_step": int(row["step_ahead"]),
        "backtest_precision": float(backtest_precision),
        "backtest_support": int(support),
        "quantile_consensus": float(q_consensus),
        "magnitude_score": float(magnitude_score),
    }


# ============================================================
# 10. SIGNAL LOGGING
# ============================================================

def append_signal_log(signal_info, context_df, signal_log_path):
    signal_log_path = Path(signal_log_path)

    last_known_timestamp = context_df["timestamp"].iloc[-1]

    row = {
        "generated_at": utc_now_naive(),
        "last_known_timestamp": last_known_timestamp,
        "target_timestamp": signal_info["target_timestamp"],
        "horizon_step": signal_info["horizon_step"],
        "signal": signal_info["signal"],
        "raw_side": signal_info["raw_side"],
        "confidence_pct": signal_info["confidence_pct"],
        "last_known_price": signal_info["last_known_price"],
        "predicted_price": signal_info["predicted_price"],
        "predicted_return_pct": signal_info["predicted_return_pct"],
        "base_threshold_pct": signal_info["base_threshold_pct"],
        "effective_threshold_pct": signal_info["effective_threshold_pct"],
        "backtest_precision": signal_info["backtest_precision"],
        "backtest_support": signal_info["backtest_support"],
        "quantile_consensus": signal_info["quantile_consensus"],
        "reason": signal_info["reason"],
    }

    new_df = pd.DataFrame([row])

    if signal_log_path.exists():
        old_df = pd.read_csv(signal_log_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(
        subset=["last_known_timestamp", "target_timestamp", "horizon_step"],
        keep="last",
    )

    combined.to_csv(signal_log_path, index=False)

    return combined


# ============================================================
# 11. PLOTTING
# ============================================================

def plot_context_and_forecast(context_df, forecast_df, qcols, chart_context_rows):
    fig, ax = plt.subplots(figsize=(12, 5))

    ctx = context_df.tail(chart_context_rows).copy()

    ax.plot(
        ctx["timestamp"],
        ctx["target"],
        label="Recent actual BTC/USD",
        linewidth=1.5,
    )

    ax.plot(
        forecast_df["target_timestamp"],
        forecast_df["predicted_price"],
        label="Forecast",
        marker="o",
        linestyle="--",
        linewidth=1.8,
    )

    # Try to plot a prediction interval if quantiles exist
    if qcols:
        q_values = [q for q, _ in qcols]

        lower_col = None
        upper_col = None

        if len(q_values) >= 2:
            lower_q, lower_col = min(qcols, key=lambda x: abs(x[0] - 0.10))
            upper_q, upper_col = min(qcols, key=lambda x: abs(x[0] - 0.90))

            if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
                ax.fill_between(
                    forecast_df["target_timestamp"],
                    forecast_df[lower_col].astype(float),
                    forecast_df[upper_col].astype(float),
                    alpha=0.2,
                    label=f"Prediction interval q{lower_q:.2f}-q{upper_q:.2f}",
                )

    ax.set_title("Recent BTC/USD Price and Chronos Forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("BTC/USD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    return fig


# ============================================================
# 12. SIDEBAR
# ============================================================

st.sidebar.header("Paths")

model_path = st.sidebar.text_input("Trained model path", DEFAULT_MODEL_PATH)

historical_csv = st.sidebar.text_input("Historical CSV/GZ", DEFAULT_HISTORICAL_CSV)

recent_csv = st.sidebar.text_input("Recent update CSV", DEFAULT_RECENT_CSV)

backtest_csv = st.sidebar.text_input("Backtest predictions CSV", DEFAULT_BACKTEST_CSV)

signal_log_path = st.sidebar.text_input("Signal log CSV", DEFAULT_SIGNAL_LOG_PATH)

git_repo_dir = st.sidebar.text_input("Git repo folder", DEFAULT_GIT_REPO_DIR)

st.sidebar.header("Data settings")

series_id = st.sidebar.text_input(
    "Series ID",
    DEFAULT_SERIES_ID,
    help="Use the same ID that you used during training.",
)

covariates_text = st.sidebar.text_input(
    "Covariate columns",
    ",".join(DEFAULT_COVARIATE_COLS),
)

covariate_cols = [c.strip() for c in covariates_text.split(",") if c.strip()]

context_window = st.sidebar.number_input(
    "Context rows",
    min_value=500,
    max_value=500000,
    value=10000,
    step=500,
)

indicator_warmup_rows = st.sidebar.number_input(
    "Indicator warmup rows",
    min_value=50,
    max_value=5000,
    value=300,
    step=50,
)

resample_rule_text = st.sidebar.text_input(
    "Resample rule",
    value="",
    help='Leave blank for 1-minute data. Example: "5min".',
)

resample_rule = resample_rule_text.strip() if resample_rule_text.strip() else None

regularize_1min = st.sidebar.checkbox(
    "Regularize missing 1-minute bars",
    value=True,
)

prefer_recent_only = st.sidebar.checkbox(
    "Use recent CSV only if it has enough rows",
    value=True,
    help="Faster for Streamlit. If recent file has enough rows for context, historical file is not loaded.",
)

run_git_pull = st.sidebar.checkbox(
    "Run git pull before refresh",
    value=False,
)

st.sidebar.header("Model settings")

preferred_model_name = st.sidebar.text_input(
    "Preferred model name",
    value="Chronos2FineTuned",
)

st.sidebar.header("Threshold / confidence settings")

fallback_threshold_pct = st.sidebar.number_input(
    "Fallback minimum move threshold (%)",
    min_value=0.0001,
    max_value=5.0,
    value=0.0300,
    step=0.005,
    format="%.5f",
)

min_threshold_pct = st.sidebar.number_input(
    "Absolute minimum threshold (%)",
    min_value=0.0001,
    max_value=5.0,
    value=0.0100,
    step=0.005,
    format="%.5f",
)

mae_multiplier = st.sidebar.number_input(
    "MAE multiplier for threshold",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

volatility_multiplier = st.sidebar.number_input(
    "Volatility multiplier for threshold",
    min_value=0.0,
    max_value=10.0,
    value=0.5,
    step=0.1,
)

target_direction_accuracy = st.sidebar.number_input(
    "Target direction accuracy",
    min_value=0.50,
    max_value=0.95,
    value=0.55,
    step=0.01,
)

min_precision_for_threshold = st.sidebar.number_input(
    "Minimum backtest precision for calibrated threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.55,
    step=0.01,
)

min_samples_for_threshold = st.sidebar.number_input(
    "Minimum backtest samples for threshold",
    min_value=1,
    max_value=500,
    value=5,
    step=1,
)

min_signal_confidence = st.sidebar.number_input(
    "Minimum live signal confidence",
    min_value=0.50,
    max_value=0.95,
    value=0.55,
    step=0.01,
)

transaction_cost_pct = st.sidebar.number_input(
    "Transaction cost / required move (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.005,
    format="%.5f",
    help="Used when deciding whether historical BUY/SELL direction was correct.",
)

confidence_threshold_min_factor = st.sidebar.number_input(
    "Min confidence threshold factor",
    min_value=0.25,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="High confidence can reduce the threshold down to this factor.",
)

confidence_threshold_max_factor = st.sidebar.number_input(
    "Max confidence threshold factor",
    min_value=1.0,
    max_value=5.0,
    value=1.75,
    step=0.05,
    help="Low confidence can increase the threshold up to this factor.",
)

chart_context_rows = st.sidebar.number_input(
    "Chart context rows",
    min_value=50,
    max_value=5000,
    value=500,
    step=50,
)

append_log = st.sidebar.checkbox(
    "Append signal to CSV log",
    value=True,
)

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)

auto_refresh_seconds = st.sidebar.number_input(
    "Auto-refresh seconds",
    min_value=10,
    max_value=3600,
    value=60,
    step=10,
)

if st.sidebar.button("Clear Streamlit cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

refresh_clicked = st.sidebar.button("Refresh forecast", type="primary")


# ============================================================
# 13. MAIN APP EXECUTION
# ============================================================

try:
    if not Path(model_path).exists():
        st.error(f"Model path does not exist: {model_path}")
        st.stop()

    with st.spinner("Loading trained AutoGluon/Chronos predictor..."):
        predictor = load_predictor_cached(model_path)

    model_names = get_model_names_safe(predictor)
    chosen_model_name = choose_model_name(predictor, preferred_model_name)

    prediction_length = int(getattr(predictor, "prediction_length", 3))

    horizon_step = st.sidebar.slider(
        "Signal horizon step",
        min_value=1,
        max_value=prediction_length,
        value=prediction_length,
        help="Which forecast step should be used for BUY/SELL/HOLD decision.",
    )

    if chosen_model_name is None:
        model_label = "AutoGluon default/best model"
    else:
        model_label = chosen_model_name

    st.subheader("Loaded Model")

    c1, c2, c3 = st.columns(3)

    c1.metric("Prediction length", prediction_length)
    c2.metric("Selected signal horizon", horizon_step)
    c3.metric("Model used", model_label)

    with st.expander("Available AutoGluon models"):
        st.write(model_names)

    # Run first time automatically
    if "has_run_once" not in st.session_state:
        st.session_state["has_run_once"] = True
        run_now = True
    else:
        run_now = refresh_clicked or auto_refresh

    if not run_now:
        st.info("Click **Refresh forecast** in the sidebar to update the signal.")
        st.stop()

    if run_git_pull:
        with st.spinner("Running git pull..."):
            maybe_git_pull(git_repo_dir)

    minimum_recent_rows = int(context_window + indicator_warmup_rows)

    with st.spinner("Loading Bitstamp dataset..."):
        df_full, source_used = load_bitstamp_dataset_cached(
            historical_csv=historical_csv,
            recent_csv=recent_csv,
            historical_mtime=get_file_mtime(historical_csv),
            recent_mtime=get_file_mtime(recent_csv),
            series_id=series_id,
            prefer_recent_only=prefer_recent_only,
            minimum_recent_rows=minimum_recent_rows,
        )

    st.subheader("Dataset Summary")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Rows loaded", f"{len(df_full):,}")
    d2.metric("Source used", source_used)
    d3.metric("First timestamp", str(df_full["timestamp"].min()))
    d4.metric("Last timestamp", str(df_full["timestamp"].max()))

    st.metric("Last close / target", f"{df_full['target'].iloc[-1]:,.2f}")

    with st.spinner("Preparing latest context and indicators..."):
        context_df = prepare_context_dataframe(
            df_full=df_full,
            series_id=series_id,
            context_window=int(context_window),
            indicator_warmup_rows=int(indicator_warmup_rows),
            covariate_cols=covariate_cols,
            resample_rule=resample_rule,
            regularize_1min=regularize_1min,
        )

    st.subheader("Context Used for Prediction")

    cc1, cc2, cc3 = st.columns(3)

    cc1.metric("Context rows", f"{len(context_df):,}")
    cc2.metric("Context first time", str(context_df["timestamp"].min()))
    cc3.metric("Context last time", str(context_df["timestamp"].max()))

    # ------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------

    with st.spinner("Running Chronos forecast..."):
        forecast_df, pred_col, qcols = predict_next_minutes(
            predictor=predictor,
            model_name=chosen_model_name,
            context_df=context_df,
            series_id=series_id,
            covariate_cols=covariate_cols,
        )

    # ------------------------------------------------------------
    # Backtest calibration
    # ------------------------------------------------------------

    backtest_df = load_backtest_csv_cached(
        backtest_csv=backtest_csv,
        backtest_mtime=get_file_mtime(backtest_csv),
    )

    calibration = build_calibration(
        backtest_df=backtest_df,
        horizon_step=horizon_step,
        fallback_threshold_pct=float(fallback_threshold_pct),
        min_threshold_pct=float(min_threshold_pct),
        mae_multiplier=float(mae_multiplier),
        volatility_multiplier=float(volatility_multiplier),
        target_direction_accuracy=float(target_direction_accuracy),
        min_precision=float(min_precision_for_threshold),
        min_samples=int(min_samples_for_threshold),
        transaction_cost_pct=float(transaction_cost_pct),
    )

    if not calibration["usable"]:
        st.warning(
            "Backtest CSV could not be used for calibration. "
            "Using fallback thresholds only. Make sure your training script saved "
            "chronos2_bitstamp_backtest_predictions.csv."
        )
    elif calibration["rows"] < 30:
        st.warning(
            f"Only {calibration['rows']} backtest rows were available for calibration. "
            "Threshold confidence may be unreliable. For better signal calibration, "
            "increase TEST_WINDOWS during backtesting."
        )

    # ------------------------------------------------------------
    # Generate signal
    # ------------------------------------------------------------

    signal_info = generate_signal(
        forecast_df=forecast_df,
        horizon_step=horizon_step,
        qcols=qcols,
        calibration=calibration,
        min_signal_confidence=float(min_signal_confidence),
        min_samples_for_confidence=int(min_samples_for_threshold),
        transaction_cost_pct=float(transaction_cost_pct),
        confidence_threshold_min_factor=float(confidence_threshold_min_factor),
        confidence_threshold_max_factor=float(confidence_threshold_max_factor),
    )

    # ------------------------------------------------------------
    # Display signal
    # ------------------------------------------------------------

    st.subheader("Trading Signal")

    signal = signal_info["signal"]

    if signal == "BUY":
        st.success(f"### Signal: BUY")
    elif signal == "SELL":
        st.error(f"### Signal: SELL")
    else:
        st.info(f"### Signal: HOLD / DO NOTHING")

    st.write(signal_info["reason"])

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Confidence", f"{signal_info['confidence_pct']:.2f}%")
    s2.metric("Predicted return", f"{signal_info['predicted_return_pct']:.5f}%")
    s3.metric("Base threshold", f"{signal_info['base_threshold_pct']:.5f}%")
    s4.metric("Effective threshold", f"{signal_info['effective_threshold_pct']:.5f}%")

    s5, s6, s7, s8 = st.columns(4)

    s5.metric("Last known price", f"{signal_info['last_known_price']:,.2f}")
    s6.metric("Predicted price", f"{signal_info['predicted_price']:,.2f}")
    s7.metric("Backtest precision", f"{signal_info['backtest_precision']:.2%}")
    s8.metric("Backtest support", signal_info["backtest_support"])

    st.caption(
        "Base threshold is calibrated from backtest MAE, volatility, and directional accuracy. "
        "Effective threshold is adjusted by current forecast confidence."
    )

    # ------------------------------------------------------------
    # Forecast table
    # ------------------------------------------------------------

    st.subheader("Minute-by-Minute Forecast")

    display_cols = [
        "target_timestamp",
        "step_ahead",
        "step_ahead_minutes",
        "last_known_price",
        "predicted_price",
        "predicted_change",
        "predicted_return_pct",
        "predicted_direction",
    ]

    q_display_cols = []

    for q, col in qcols:
        if q in [0.1, 0.2, 0.5, 0.8, 0.9] or len(qcols) <= 7:
            new_name = f"q{q:.2f}"
            forecast_df[new_name] = forecast_df[col]
            q_display_cols.append(new_name)

    display_df = forecast_df[display_cols + q_display_cols].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------
    # Calibration metrics
    # ------------------------------------------------------------

    st.subheader("Training / Backtest Calibration Performance")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Calibration rows", calibration["rows"])
    p2.metric(
        "Backtest MAE %",
        "N/A" if not np.isfinite(calibration["mae_pct"]) else f"{calibration['mae_pct']:.5f}%",
    )
    p3.metric(
        "Backtest RMSE %",
        "N/A" if not np.isfinite(calibration["rmse_pct"]) else f"{calibration['rmse_pct']:.5f}%",
    )
    p4.metric(
        "Direction accuracy",
        "N/A" if not np.isfinite(calibration["trend_accuracy"]) else f"{calibration['trend_accuracy']:.2%}",
    )

    t1, t2, t3 = st.columns(3)

    t1.metric(
        "Dynamic base threshold",
        f"{calibration['base_dynamic_threshold_pct']:.5f}%",
    )

    t2.metric(
        "BUY threshold",
        f"{calibration['buy_threshold_pct']:.5f}%",
    )

    t3.metric(
        "SELL threshold",
        f"{calibration['sell_threshold_pct']:.5f}%",
    )

    threshold_details = pd.DataFrame(
        [
            {
                "side": "BUY",
                "threshold_pct": calibration["buy_threshold_pct"],
                "precision": calibration["buy_stats"].get("precision", np.nan),
                "support": calibration["buy_stats"].get("support", 0),
                "source": calibration["buy_stats"].get("source", ""),
            },
            {
                "side": "SELL",
                "threshold_pct": calibration["sell_threshold_pct"],
                "precision": calibration["sell_stats"].get("precision", np.nan),
                "support": calibration["sell_stats"].get("support", 0),
                "source": calibration["sell_stats"].get("source", ""),
            },
        ]
    )

    st.dataframe(threshold_details, use_container_width=True, hide_index=True)

    with st.expander("Backtest rows used for calibration"):
        if calibration["usable"]:
            st.dataframe(
                calibration["calibration_df"].tail(200),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No usable calibration rows.")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    st.subheader("Price and Forecast Chart")

    fig = plot_context_and_forecast(
        context_df=context_df,
        forecast_df=forecast_df,
        qcols=qcols,
        chart_context_rows=int(chart_context_rows),
    )

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Append signal log
    # ------------------------------------------------------------

    if append_log:
        log_df = append_signal_log(
            signal_info=signal_info,
            context_df=context_df,
            signal_log_path=signal_log_path,
        )

        st.success(f"Signal saved to: {signal_log_path}")

        with st.expander("Latest signal log rows"):
            st.dataframe(log_df.tail(50), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------

    if auto_refresh:
        st.info(f"Auto-refreshing in {auto_refresh_seconds} seconds...")
        time.sleep(int(auto_refresh_seconds))
        st.rerun()

except Exception as e:
    st.exception(e)
