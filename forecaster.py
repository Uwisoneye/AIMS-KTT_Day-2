import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
ARTIFACT_DIR = ROOT / 'artifacts'
MODEL_DIR = ROOT / 'models'
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    'hour', 'dow', 'is_weekend',
    'load_lag24', 'load_lag48', 'load_lag168',
    'temp_lag24', 'humidity_lag24', 'wind_lag24',
    'rain_lag24', 'rain_lag48', 'rain_lag168',
    'outage_lag24', 'duration_lag24',
    'rain_roll24_lag24', 'outage_rate_samehour_prev7d',
    'load_diff_24_168'
]


def load_history(path: Path = DATA_DIR / 'grid_history.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values('timestamp').reset_index(drop=True)
    x['hour'] = x['timestamp'].dt.hour
    x['dow'] = x['timestamp'].dt.dayofweek
    x['is_weekend'] = (x['dow'] >= 5).astype(int)

    for lag in [24, 48, 168]:
        x[f'load_lag{lag}'] = x['load_mw'].shift(lag)
        x[f'rain_lag{lag}'] = x['rain_mm'].shift(lag)
    x['temp_lag24'] = x['temp_c'].shift(24)
    x['humidity_lag24'] = x['humidity'].shift(24)
    x['wind_lag24'] = x['wind_ms'].shift(24)
    x['outage_lag24'] = x['outage'].shift(24)
    x['duration_lag24'] = x['duration_min'].shift(24)
    x['rain_roll24_lag24'] = x['rain_mm'].rolling(24).mean().shift(24)
    x['outage_rate_samehour_prev7d'] = x['outage'].shift(24).rolling(7 * 24, min_periods=24).mean()
    x['load_diff_24_168'] = x['load_lag24'] - x['load_lag168']

    x = x.dropna().reset_index(drop=True)
    return x


def build_classifier() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])


def build_duration_regressor() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])


def fit_models(train_df: pd.DataFrame) -> Tuple[Pipeline, Pipeline]:
    clf = build_classifier()
    clf.fit(train_df[FEATURES], train_df['outage'])

    pos = train_df[train_df['outage'] == 1].copy()
    reg = build_duration_regressor()
    if len(pos) < 10:
        pos = train_df.nlargest(10, 'duration_min').copy()
    reg.fit(pos[FEATURES], np.log1p(pos['duration_min']))
    return clf, reg


def predict_frame(df: pd.DataFrame, clf: Pipeline, reg: Pipeline) -> pd.DataFrame:
    out = df[['timestamp', 'outage', 'duration_min']].copy()
    out['p_outage'] = clf.predict_proba(df[FEATURES])[:, 1]
    out['expected_duration_if_outage_min'] = np.expm1(reg.predict(df[FEATURES])).clip(15, 360)
    return out


def choose_threshold(pred: np.ndarray, y_true: np.ndarray) -> float:
    candidates = np.linspace(0.03, 0.2, 35)
    best_t, best_f1 = 0.4, -1.0
    for t in candidates:
        score = f1_score(y_true, (pred >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t


def bootstrap_band(train_df: pd.DataFrame, future_df: pd.DataFrame, n_boot: int = 7) -> pd.DataFrame:
    probs = []
    for i in range(n_boot):
        sample = train_df.sample(frac=1.0, replace=True, random_state=100 + i)
        clf, _ = fit_models(sample)
        probs.append(clf.predict_proba(future_df[FEATURES])[:, 1])
    arr = np.vstack(probs)
    return pd.DataFrame({
        'p_lower': np.quantile(arr, 0.10, axis=0),
        'p_upper': np.quantile(arr, 0.90, axis=0),
    })


def rolling_evaluate(feat_df: pd.DataFrame, holdout_days: int = 30) -> Tuple[Dict, pd.DataFrame]:
    end_time = feat_df['timestamp'].max()
    holdout_start = (end_time - pd.Timedelta(days=holdout_days)) + pd.Timedelta(hours=1)
    start_day = holdout_start.normalize()
    end_day = end_time.normalize()

    daily_predictions = []
    pre_holdout = feat_df[feat_df['timestamp'] < start_day].copy()
    base_clf, _ = fit_models(pre_holdout)
    base_probs = base_clf.predict_proba(pre_holdout[FEATURES])[:, 1]
    threshold = choose_threshold(base_probs, pre_holdout['outage'].values)

    day = start_day
    while day <= end_day:
        train = feat_df[feat_df['timestamp'] < day].copy()
        test = feat_df[(feat_df['timestamp'] >= day) & (feat_df['timestamp'] < day + pd.Timedelta(hours=24))].copy()
        if len(train) < 500 or len(test) == 0:
            day += pd.Timedelta(days=1)
            continue
        clf, reg = fit_models(train)
        pred = predict_frame(test, clf, reg)
        pred['forecast_day'] = day
        pred['forecast_hour_ahead'] = ((pred['timestamp'] - day).dt.total_seconds() / 3600).astype(int) + 1
        pred['alert'] = (pred['p_outage'] >= threshold).astype(int)
        daily_predictions.append(pred)
        day += pd.Timedelta(days=1)

    pred_df = pd.concat(daily_predictions, ignore_index=True)
    outage_mask = pred_df['outage'] == 1
    mae = float(mean_absolute_error(pred_df.loc[outage_mask, 'duration_min'], pred_df.loc[outage_mask, 'expected_duration_if_outage_min'])) if outage_mask.any() else None
    lead_time = float(pred_df.loc[outage_mask & (pred_df['alert'] == 1), 'forecast_hour_ahead'].mean()) if (outage_mask & (pred_df['alert'] == 1)).any() else 0.0

    metrics = {
        'brier_score': float(brier_score_loss(pred_df['outage'], pred_df['p_outage'])),
        'duration_mae_on_true_outages_min': mae,
        'average_lead_time_hours_on_true_outages': lead_time,
        'threshold': threshold,
        'holdout_days': holdout_days,
        'holdout_rows': int(len(pred_df)),
        'outage_rate_holdout': float(pred_df['outage'].mean()),
    }
    return metrics, pred_df


def train_final_and_forecast(feat_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Pipeline, Pipeline]:
    cutoff = feat_df['timestamp'].max().normalize()
    train = feat_df[feat_df['timestamp'] < cutoff].copy()
    future = feat_df[feat_df['timestamp'] >= cutoff].copy().head(24)
    clf, reg = fit_models(train)

    probs = clf.predict_proba(train[FEATURES])[:, 1]
    threshold = choose_threshold(probs, train['outage'].values)

    forecast = predict_frame(future, clf, reg)
    band = bootstrap_band(train, future)
    forecast = pd.concat([forecast.reset_index(drop=True), band], axis=1)
    forecast['threshold'] = threshold
    forecast['hour'] = forecast['timestamp'].dt.strftime('%H:%M')

    coef = clf.named_steps['model'].coef_[0]
    importance = pd.DataFrame({'feature': FEATURES, 'coef': coef, 'abs_coef': np.abs(coef)})
    importance = importance.sort_values('abs_coef', ascending=False).reset_index(drop=True)

    meta = {
        'threshold': threshold,
        'top_features': importance.head(5).to_dict(orient='records')
    }
    return meta, forecast, importance, clf, reg


def save_artifacts(metrics: Dict, pred_df: pd.DataFrame, forecast_meta: Dict, forecast_df: pd.DataFrame, importance: pd.DataFrame, clf: Pipeline, reg: Pipeline) -> None:
    (ARTIFACT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pred_df.to_csv(ARTIFACT_DIR / 'eval_predictions.csv', index=False)
    forecast_df.to_csv(ARTIFACT_DIR / 'today_forecast.csv', index=False)
    importance.to_csv(ARTIFACT_DIR / 'feature_importance.csv', index=False)
    (ARTIFACT_DIR / 'forecast_meta.json').write_text(json.dumps(forecast_meta, indent=2), encoding='utf-8')
    with open(MODEL_DIR / 'outage_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(MODEL_DIR / 'duration_regressor.pkl', 'wb') as f:
        pickle.dump(reg, f)


def run_all() -> Dict:
    raw = load_history()
    feat = make_features(raw)
    metrics, pred_df = rolling_evaluate(feat)
    meta, forecast_df, importance, clf, reg = train_final_and_forecast(feat)
    metrics.update(meta)
    save_artifacts(metrics, pred_df, meta, forecast_df, importance, clf, reg)
    return metrics


if __name__ == '__main__':
    print(json.dumps(run_all(), indent=2))
