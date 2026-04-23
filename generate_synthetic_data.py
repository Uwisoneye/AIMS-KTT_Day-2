import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_grid_history():
    end = pd.Timestamp('2026-04-23 23:00:00')
    periods = 180 * 24
    ts = pd.date_range(end=end, periods=periods, freq='h')
    hour = ts.hour.values
    dow = ts.dayofweek.values
    doy = ts.dayofyear.values

    morning_peak = 10 * np.exp(-0.5 * ((hour - 9) / 2.4) ** 2)
    evening_peak = 18 * np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
    weekly = np.where(dow < 5, 3.5, -2.0)
    seasonal = 2.0 * np.sin(2 * np.pi * doy / 365.0)

    temp = 23 + 4 * np.sin(2 * np.pi * (hour - 14) / 24) + 1.8 * np.sin(2 * np.pi * doy / 365.0) + rng.normal(0, 1.2, periods)
    humidity = np.clip(68 - 0.6 * temp + 15 * rng.random(periods), 35, 98)
    wind = np.clip(2.2 + 1.5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.6, periods), 0.1, None)

    rainy_season = ((ts.month >= 2) & (ts.month <= 5)).astype(float)
    rain_prob = 0.12 + 0.24 * rainy_season + 0.05 * np.isin(hour, [13, 14, 15, 16]).astype(float)
    rain = np.where(rng.random(periods) < rain_prob, rng.gamma(shape=1.8 + rainy_season, scale=2.5, size=periods), 0.0)

    load = 52 + morning_peak + evening_peak + weekly + seasonal + 0.10 * humidity + 0.18 * temp + 0.30 * rain + rng.normal(0, 2.5, periods)

    load_lag1 = np.roll(load, 1)
    load_lag1[0] = load[0]
    logit = -5.7 + 0.032 * load_lag1 + 0.050 * rain + 0.22 * np.isin(hour, [18, 19, 20, 21]).astype(float) + 0.10 * rainy_season
    p_out = sigmoid(logit)
    outage = rng.binomial(1, p_out)

    duration = np.zeros(periods)
    mask = outage == 1
    duration[mask] = np.clip(rng.lognormal(mean=np.log(72), sigma=0.6, size=mask.sum()), 15, 420)

    df = pd.DataFrame({
        'timestamp': ts,
        'load_mw': np.round(load, 3),
        'temp_c': np.round(temp, 2),
        'humidity': np.round(humidity, 2),
        'wind_ms': np.round(wind, 2),
        'rain_mm': np.round(rain, 2),
        'outage': outage.astype(int),
        'duration_min': np.round(duration, 1),
    })
    return df


def generate_appliances():
    appliances = [
        {'name': 'lights', 'category': 'critical', 'watts_avg': 180, 'start_up_spike_w': 220, 'revenue_if_running_rwf_per_h': 3500},
        {'name': 'phone_charging', 'category': 'critical', 'watts_avg': 120, 'start_up_spike_w': 150, 'revenue_if_running_rwf_per_h': 1800},
        {'name': 'cash_system', 'category': 'critical', 'watts_avg': 90, 'start_up_spike_w': 120, 'revenue_if_running_rwf_per_h': 2200},
        {'name': 'clipper_station', 'category': 'comfort', 'watts_avg': 240, 'start_up_spike_w': 320, 'revenue_if_running_rwf_per_h': 6000},
        {'name': 'hair_dryer', 'category': 'luxury', 'watts_avg': 1500, 'start_up_spike_w': 1900, 'revenue_if_running_rwf_per_h': 7000},
        {'name': 'water_heater', 'category': 'luxury', 'watts_avg': 1800, 'start_up_spike_w': 2200, 'revenue_if_running_rwf_per_h': 5000},
        {'name': 'freezer', 'category': 'critical', 'watts_avg': 900, 'start_up_spike_w': 1800, 'revenue_if_running_rwf_per_h': 12000},
        {'name': 'cold_room_fan', 'category': 'critical', 'watts_avg': 650, 'start_up_spike_w': 900, 'revenue_if_running_rwf_per_h': 8000},
        {'name': 'sewing_machine', 'category': 'comfort', 'watts_avg': 350, 'start_up_spike_w': 500, 'revenue_if_running_rwf_per_h': 6500},
        {'name': 'iron_press', 'category': 'luxury', 'watts_avg': 1200, 'start_up_spike_w': 1600, 'revenue_if_running_rwf_per_h': 5500},
    ]
    return appliances


def generate_businesses():
    businesses = {
        'salon': ['lights', 'phone_charging', 'cash_system', 'clipper_station', 'hair_dryer', 'water_heater'],
        'cold_room': ['lights', 'phone_charging', 'cash_system', 'freezer', 'cold_room_fan'],
        'tailor': ['lights', 'phone_charging', 'cash_system', 'sewing_machine', 'iron_press'],
    }
    return businesses


def main():
    grid = generate_grid_history()
    appliances = generate_appliances()
    businesses = generate_businesses()

    grid.to_csv(DATA_DIR / 'grid_history.csv', index=False)
    with open(DATA_DIR / 'appliances.json', 'w', encoding='utf-8') as f:
        json.dump(appliances, f, indent=2)
    with open(DATA_DIR / 'businesses.json', 'w', encoding='utf-8') as f:
        json.dump(businesses, f, indent=2)

    summary = {
        'rows': int(len(grid)),
        'outage_rate': float(grid['outage'].mean()),
        'mean_duration_min_when_outage': float(grid.loc[grid['outage'] == 1, 'duration_min'].mean()),
        'appliance_count': len(appliances),
        'businesses': list(businesses.keys()),
        'seed': SEED,
    }
    with open(DATA_DIR / 'data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
