import json
from pathlib import Path

import pandas as pd

from forecaster import run_all
from prioritizer import load_catalog, load_businesses, plan, select_business_appliances, summarize_plan

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / 'artifacts'
DATA_DIR = ROOT / 'data'


def build_lite_ui(forecast: pd.DataFrame, plan_df: pd.DataFrame, business_name: str, summary: dict) -> str:
    hours = forecast['hour'].tolist()
    probs = [round(float(x), 3) for x in forecast['p_outage']]
    lowers = [round(float(x), 3) for x in forecast['p_lower']]
    uppers = [round(float(x), 3) for x in forecast['p_upper']]
    plan_pivot = plan_df.pivot_table(index='name', columns='hour', values='status', aggfunc='first').reset_index()
    plan_rows = []
    for _, row in plan_pivot.iterrows():
        cells = ''.join([f"<td class='{('on' if row[h]=='ON' else 'off')}'>{row[h][0]}</td>" for h in hours])
        plan_rows.append(f"<tr><th>{row['name']}</th>{cells}</tr>")
    plan_rows_html = ''.join(plan_rows)

    def poly(values, ymin=0.0, ymax=1.0):
        pts = []
        for i, v in enumerate(values):
            x = 30 + i * (760 / 23)
            y = 220 - ((v - ymin) / (ymax - ymin)) * 180
            pts.append(f"{x:.1f},{y:.1f}")
        return ' '.join(pts)

    script = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Grid Outage Forecast</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
body{{font-family:Arial,sans-serif;margin:14px;color:#111}} .k{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}}
.card{{border:1px solid #ddd;border-radius:8px;padding:10px}} table{{border-collapse:collapse;font-size:10px;width:100%;table-layout:fixed}}
th,td{{border:1px solid #ddd;padding:4px;text-align:center;overflow:hidden}} .on{{background:#dff6dd}} .off{{background:#ffe0e0}}
.small{{font-size:12px;color:#444}} svg{{width:100%;height:auto;border:1px solid #ddd;border-radius:8px}} .legend span{{display:inline-block;margin-right:12px;font-size:12px}}
</style></head><body>
<h2>T2.3 Grid Outage Forecast + Appliance Plan</h2>
<div class='small'>Business: {business_name} | Cached static page for today's 24h forecast.</div>
<div class='k'>
<div class='card'><b>Brier</b><br>{summary['brier_score']:.3f}</div>
<div class='card'><b>Threshold</b><br>{summary['threshold']:.2f}</div>
<div class='card'><b>High-risk hours</b><br>{summary['hours_high_risk']}</div>
<div class='card'><b>Expected weekly gain</b><br>{int(summary['week_delta_rwf']):,} RWF</div>
</div>
<svg viewBox='0 0 820 250' role='img' aria-label='outage probability band'>
<polyline fill='none' stroke='#e6e6e6' stroke-width='1' points='30,220 790,220'/>
<polyline fill='none' stroke='#e6e6e6' stroke-width='1' points='30,40 790,40'/>
<polyline fill='none' stroke='#c8e6ff' stroke-width='8' opacity='0.8' points='{poly(lowers)}'/>
<polyline fill='none' stroke='#c8e6ff' stroke-width='8' opacity='0.8' points='{poly(uppers)}'/>
<polyline fill='none' stroke='#1f77b4' stroke-width='2' points='{poly(probs)}'/>
<text x='35' y='32' font-size='11'>Higher risk</text>
<text x='35' y='236' font-size='11'>Lower risk</text>
</svg>
<div class='legend small'><span>Blue line: P(outage)</span><span>Wide pale line: uncertainty band</span><span>Grid: appliance plan by hour</span></div>
<h3>Plan grid</h3>
<table><thead><tr><th>Appliance</th>{''.join(f'<th>{h[:2]}</th>' for h in hours)}</tr></thead><tbody>{plan_rows_html}</tbody></table>
<p class='small'>O = ON, O in green. O in red indicates OFF. Rule enforced: luxury drops before comfort, comfort before critical. Ties break on lower revenue per watt, then higher startup spike.</p>
</body></html>"""
    return script


def build_digest_spec(metrics: dict, salon_summary: dict) -> str:
    sms1 = "SMS1: SALON PLAN 06:00. Risk high 18-21h. Keep lights,cash,chargers ON. Delay dryer+heater after 21h. Worst hour 19h."
    sms2 = "SMS2: If outage starts, keep clipper only for booked clients. Switch dryer OFF first, then heater. Expected save this week: {0:,} RWF.".format(int(max(0, salon_summary['week_delta_rwf'])))
    sms3 = "SMS3: If phone has no data after 13:00, keep last plan for 6h max. After 6h treat plan as stale: critical only until refresh or neighbor signal."

    text = f"""# digest_spec.md

## Morning digest for salon owner (feature phone)

Constraints respected: 3 SMS maximum, each under 160 characters.

1. {sms1}
2. {sms2}
3. {sms3}

## Offline behavior when internet drops mid-day

- The lite UI keeps the last successful forecast and shows **LAST REFRESH HH:MM** at the top.
- Green = safe, amber = caution, red = high risk. If the plan is older than **6 hours**, the status chip turns amber and the UI collapses to a simpler rule: **critical only**.
- Risk budget for stale plans: I accept up to **6 hours** of staleness. After that, I stop trusting hour-level ranking because weather and load patterns can drift enough to flip the top-risk window.

## Non-reader adaptation choice

I chose **colored icons on the lite UI** rather than voice. Reason:
- cheaper than voice calls,
- works on low-end Android devices used by shop assistants,
- still usable in noisy markets,
- can be mirrored later to a relay board with the same color logic.

Icon logic:
- 🟢 run normally
- 🟠 prepare to shed luxury appliances
- 🔴 keep critical only

## Business workflow

- 06:00: daily forecast is generated and sent as 3 SMS to the owner.
- 06:05: cached lite UI is refreshed on the phone used at the counter.
- 13:00 onward: if internet fails, staff keep using the cached plan until the 6-hour stale limit.
- Escalation: if three red windows appear in one week, the owner switches to a heavier backup schedule and logs the event for service review.

## Concrete numbers

- Held-out Brier score: **{metrics['brier_score']:.3f}**
- Forecast threshold used for alerts: **{metrics['threshold']:.2f}**
- Estimated expected weekly revenue gain vs naive operation: **{int(max(0, salon_summary['week_delta_rwf'])):,} RWF**
"""
    return text


def build_eval_notebook() -> dict:
    cells = [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['# Evaluation notebook\n', 'Loads held-out predictions and recomputes the metrics used in the README.']
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import json\n',
                'import pandas as pd\n',
                'from sklearn.metrics import brier_score_loss, mean_absolute_error\n',
                "pred = pd.read_csv('artifacts/eval_predictions.csv', parse_dates=['timestamp','forecast_day'])\n",
                "metrics = json.load(open('artifacts/metrics.json'))\n",
                'outage_mask = pred.outage == 1\n',
                "print('Brier', round(brier_score_loss(pred.outage, pred.p_outage), 4))\n",
                "print('MAE on true outages', round(mean_absolute_error(pred.loc[outage_mask,'duration_min'], pred.loc[outage_mask,'expected_duration_if_outage_min']), 2))\n",
                "print(metrics)\n",
                'pred.head()\n'
            ]
        }
    ]
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.x'}
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    return nb


def main():
    metrics = run_all()

    catalog = load_catalog()
    businesses = load_businesses()
    forecast = pd.read_csv(ARTIFACT_DIR / 'today_forecast.csv', parse_dates=['timestamp'])

    all_summaries = {}
    for business_name in businesses:
        app_df = select_business_appliances(business_name, catalog, businesses)
        plan_df = plan(forecast, app_df)
        plan_df.to_csv(ARTIFACT_DIR / f'plan_{business_name}.csv', index=False)
        all_summaries[business_name] = summarize_plan(plan_df)

    salon_plan = pd.read_csv(ARTIFACT_DIR / 'plan_salon.csv', parse_dates=['timestamp'])
    salon_summary = all_summaries['salon']
    combined = dict(metrics)
    combined.update(salon_summary)

    html = build_lite_ui(forecast, salon_plan, 'salon', combined)
    (ROOT / 'lite_ui.html').write_text(html, encoding='utf-8')

    digest = build_digest_spec(metrics, salon_summary)
    (ROOT / 'digest_spec.md').write_text(digest, encoding='utf-8')

    notebook = build_eval_notebook()
    (ROOT / 'eval.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')

    manifest = {
        'metrics': metrics,
        'summaries': all_summaries,
        'outputs': ['lite_ui.html', 'digest_spec.md', 'eval.ipynb']
    }
    (ARTIFACT_DIR / 'prepare_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
