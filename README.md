# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/JackEasons/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/JackEasons/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI trophy's Exophase earn rate a week after launch, as five quantiles on the logit scale, from an earners/players decomposition with refresh-aware spreads.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

An earn rate is a ratio, `earners / players_tracked`, and the two sides move
for different reasons, so `main.py` (Python 3.13, standard library only,
offline, deterministic) forecasts them separately and divides:

- **Players tracked** grows in jumps whenever Exophase refreshes a game's
  counts (hours apart), not continuously. Between refreshes the page is
  frozen.
- **Earners** for a trophy grow only as players actually reach it. Early
  chapters and quick activities accumulate earners fast; late, grind and
  completion trophies barely move at first. Earners are fitted per trophy
  on a log-time scale, with slopes shrunk toward the cross-trophy median
  and recent refreshes weighted more than the launch-night plateau.

Two regimes, chosen from the case's own history:

1. **Trend** (two or more distinct snapshots): log-time fits for players
   and for each trophy's earners and logit rate, read at the target.
2. **Launch night** (a single frozen snapshot, as in the practice cases):
   the documented first refresh is expected — players 638 → 3,868,
   earners 305 → 408 excluding the trophy every tracked player holds —
   with the new earners allocated by an "earliness" weight built from each
   trophy's description.

The spread combines the fits' residuals with a random-walk term that grows
with the horizon, so the five quantiles are an honest statement of how sure
the forecast is, not just a point estimate with padding.

---

<sub>Entered on [make.fun](https://make.fun/?ref=steven-yi&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
