import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
ARTIFACT_DIR = ROOT / 'artifacts'

CATEGORY_KEEP_ORDER = ['critical', 'comfort', 'luxury']
CATEGORY_DROP_ORDER = ['luxury', 'comfort', 'critical']
CATEGORY_RANK = {'critical': 0, 'comfort': 1, 'luxury': 2}


def load_catalog() -> pd.DataFrame:
    with open(DATA_DIR / 'appliances.json', 'r', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))


def load_businesses() -> Dict[str, List[str]]:
    with open(DATA_DIR / 'businesses.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def select_business_appliances(name: str, catalog: pd.DataFrame, businesses: Dict[str, List[str]]) -> pd.DataFrame:
    names = businesses[name]
    return catalog[catalog['name'].isin(names)].copy().reset_index(drop=True)


def compute_power_cap(total_watts: float, p_outage: float, expected_duration_if_outage_min: float) -> float:
    if p_outage < 0.07:
        return total_watts
    duration_factor = min(expected_duration_if_outage_min / 180.0, 1.0)
    risk_factor = np.clip(1.0 - (1.6 * (p_outage - 0.07) + 0.30 * duration_factor), 0.45, 1.0)
    return total_watts * risk_factor


def plan(forecast: pd.DataFrame, appliances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_watts = float(appliances['watts_avg'].sum())
    naive_revenue = float(appliances['revenue_if_running_rwf_per_h'].sum())

    for _, f in forecast.iterrows():
        cap = compute_power_cap(total_watts, float(f['p_outage']), float(f['expected_duration_if_outage_min']))
        current = appliances.copy()
        current['status'] = 'ON'
        current['revenue_density'] = current['revenue_if_running_rwf_per_h'] / current['watts_avg']

        running_watts = float(current['watts_avg'].sum())
        if running_watts > cap:
            for category in CATEGORY_DROP_ORDER:
                idx = current[current['category'] == category].sort_values(
                    ['revenue_density', 'start_up_spike_w', 'revenue_if_running_rwf_per_h'],
                    ascending=[True, False, True]
                ).index.tolist()
                for i in idx:
                    if running_watts <= cap:
                        break
                    current.loc[i, 'status'] = 'OFF'
                    running_watts -= float(current.loc[i, 'watts_avg'])
                if running_watts <= cap:
                    break

        overload_ratio = max(0.0, total_watts / max(cap, 1.0) - 1.0)
        if float(f['p_outage']) < 0.07:
            naive_delivery_factor = 1.0
            planned_delivery_factor = 1.0
        else:
            naive_delivery_factor = max(0.10, 1.0 - 10.0 * float(f['p_outage']) * max(overload_ratio, 0.40))
            planned_delivery_factor = max(0.90, 1.0 - 0.05 * float(f['p_outage']))

        current['timestamp'] = f['timestamp']
        current['hour'] = pd.to_datetime(f['timestamp']).strftime('%H:%M')
        current['p_outage'] = float(f['p_outage'])
        current['p_lower'] = float(f.get('p_lower', max(0.0, f['p_outage'] - 0.1)))
        current['p_upper'] = float(f.get('p_upper', min(1.0, f['p_outage'] + 0.1)))
        current['expected_duration_if_outage_min'] = float(f['expected_duration_if_outage_min'])
        current['power_cap_w'] = round(cap, 1)
        current['naive_full_on_revenue_rwf'] = naive_revenue * naive_delivery_factor
        current['planned_revenue_rwf'] = np.where(current['status'] == 'ON', current['revenue_if_running_rwf_per_h'] * planned_delivery_factor, 0)
        current['priority_rank'] = current['category'].map(CATEGORY_RANK)
        rows.append(current)

    plan_df = pd.concat(rows, ignore_index=True)
    return plan_df


def summarize_plan(plan_df: pd.DataFrame) -> Dict:
    hourly = plan_df.groupby('timestamp').agg(
        naive_revenue_rwf=('naive_full_on_revenue_rwf', 'max'),
        planned_revenue_rwf=('planned_revenue_rwf', 'sum'),
        running_watts=('watts_avg', lambda s: float(s[plan_df.loc[s.index, 'status'] == 'ON'].sum())),
        power_cap_w=('power_cap_w', 'max'),
        p_outage=('p_outage', 'max'),
    ).reset_index()
    hourly['revenue_saved_vs_naive_rwf'] = hourly['planned_revenue_rwf'] - hourly['naive_revenue_rwf']
    return {
        'naive_daily_revenue_rwf': float(hourly['naive_revenue_rwf'].sum()),
        'planned_daily_revenue_rwf': float(hourly['planned_revenue_rwf'].sum()),
        'daily_delta_rwf': float(hourly['revenue_saved_vs_naive_rwf'].sum()),
        'hours_high_risk': int((hourly['p_outage'] >= 0.07).sum()),
        'week_delta_rwf': float(hourly['revenue_saved_vs_naive_rwf'].sum() * 7),
    }


if __name__ == '__main__':
    catalog = load_catalog()
    businesses = load_businesses()
    forecast = pd.read_csv(ARTIFACT_DIR / 'today_forecast.csv', parse_dates=['timestamp'])
    salon = select_business_appliances('salon', catalog, businesses)
    result = plan(forecast, salon)
    print(result.head())
