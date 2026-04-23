# digest_spec.md

## Morning digest for salon owner (feature phone)

Constraints respected: 3 SMS maximum, each under 160 characters.

1. SMS1: SALON PLAN 06:00. Risk high 18-21h. Keep lights,cash,chargers ON. Delay dryer+heater after 21h. Worst hour 19h.
2. SMS2: If outage starts, keep clipper only for booked clients. Switch dryer OFF first, then heater. Expected save this week: 80,366 RWF.
3. SMS3: If phone has no data after 13:00, keep last plan for 6h max. After 6h treat plan as stale: critical only until refresh or neighbor signal.

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

- Held-out Brier score: **0.039**
- Forecast threshold used for alerts: **0.06**
- Estimated expected weekly revenue gain vs naive operation: **80,366 RWF**
