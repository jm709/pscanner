# Known wallet clusters

Referenced from `CLAUDE.md`. Read this when doing wallet-cluster research, watchlist curation,
or interpreting paper-trading cohort anomalies.

Primary investigation log: `volume-farming-cluster-investigation.md` covers the
two non-Cavill operations with full methodology + reproducible queries.

## Cavill cluster (manually discovered 2026-04-25)

9-wallet coordinated operation, all created **Feb 20-21 2026** (7 of 9 within a 38-minute window on Feb 20). Bimodal trade sizing ($500-999 chunks just below $1K + sub-$100 dust), 35-40% SELL rate, 57% of trades at price ≥0.95 (BUY-NO spread harvest), $0 net exposure across all 9 wallets. Behavior: market-making / Polymarket maker-rebate farming on niche long-tail markets (Henry Cavill James Bond, Cabello as Venezuelan leader, Ferran Torres top La Liga scorer, Houston Dynamo MLS Cup, Mohammad Khatami, Manchester United 2nd place EPL). Useful as a "fastest-reactor-to-mispricings" signal, NOT as an "informed insider" signal.

```
0x5cbd326a7f9dfac9855b9a23caee48fc097eabb0
0x53daff4663382b86808feb77e4fcaffd94e57cc8
0x13b775f8a46762d031cbf9a6a478fe90a81e0aaf
0x7bfbc1e83ffb9203b29f653e5367acd3a580f6f8
0xd5983aab43ef59620fda70599e30e693fd93c659
0x43d621fc31491eec23d9f696dcfb7e8923cd8ac9
0xcbd11366479deef70576a4c7c0f6eda1bc6aed42
0xf04e089482c1349d3556a36951b033094731b79b
0x5266edffc8f4737c2b9d0fa959ecae2c7b55c8cb
```

Re-add to watchlist after a DB reset:
```bash
for a in 0x5cbd326a7f9dfac9855b9a23caee48fc097eabb0 0x53daff4663382b86808feb77e4fcaffd94e57cc8 \
         0x13b775f8a46762d031cbf9a6a478fe90a81e0aaf 0x7bfbc1e83ffb9203b29f653e5367acd3a580f6f8 \
         0xd5983aab43ef59620fda70599e30e693fd93c659 0x43d621fc31491eec23d9f696dcfb7e8923cd8ac9 \
         0xcbd11366479deef70576a4c7c0f6eda1bc6aed42 0xf04e089482c1349d3556a36951b033094731b79b \
         0x5266edffc8f4737c2b9d0fa959ecae2c7b55c8cb; do
  uv run pscanner watch "$a" --reason cavill-cluster-feb2026
done
```

## Volume-farming cluster (722+ wallets, Feb-Apr 2026)
Discovered 2026-04-27 during paper-trading data exploration. Sub-$10 dust trades, 9% sell rate, gradually accumulated Feb-Apr 2026 (96% in April), 7.1% WR, **−23.8% ROI** on $6.35M. Almost certainly Polymarket points/airdrop farming. Full investigation in `volume-farming-cluster-investigation.md`.

## Magic / long-shot cluster (17 strict + ~700 fresh, Mar-Apr 2026)
Surfaced 2026-04-28 during expanded-paper-trading smoke. 50-day creation span, 47% Mondays, 6 wallets created in the burst day. **Buy YES at sub-$0.05 on tail outcomes**, hold to resolution. 14.6% WR, **−34.6% ROI** on $1.1M. Different fingerprint from volume-farming (mid-range $25 trade size vs sub-$10 dust). Same airdrop-farming end state. Full investigation in `volume-farming-cluster-investigation.md` (Update 2026-04-28 section).
