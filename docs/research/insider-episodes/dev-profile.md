# Dev-split insider wallet profile

Date: 2026-10-11. Throwaway exploratory probe, dev cases only, per the spec
`docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md`
("Exploratory profile"). No holdout case was opened. Source: desktop corpus
(indexed lookups only) plus Polymarket data-api `/activity`.

Purpose: look at what the published dev-case wallets actually did, side by
side with the crowd that bought the same side in the same 72-hour pre-news
window, so the scorer is designed from evidence rather than intuition.

## Wallets examined

| Wallet | Case | Corpus trades | Corpus markets | First corpus trade | First data-api activity | Days before news |
|---|---|---:|---:|---|---|---:|
| `0x0afc7ce5` ricosuave666 | Israel strike, 2025-06-13 00:00 | 100 | 5 | 2025-06-10 17:05 | 2025-06-10 17:05 | 2.3 |
| `0x1caa6a7a` | US-Israel strikes, 2026-02-28 06:15 | 80 | 6 | 2026-02-25 21:40 | 2026-02-25 21:40 | 2.4 |
| `0x45647817` | US-Israel strikes, 2026-02-28 06:15 | 27 | 15 | 2026-02-18 20:44 | 2024-11-05 22:48 | 9.4 (one Harris bet 16 months earlier, then dormant) |
| `0xefd06d1a` | Jan 2026 negative example | 67 | 17 | 2026-01-07 16:33 | 2026-01-07 16:33 | n/a |

All three positive wallets were fresh: their first trade in the corpus and
their first activity on the data-api fall within ten days of the event, and
for two of three within two and a half days. The data-api and the corpus agree
on first activity for the two fresh wallets, so the corpus "first trade"
is a usable age proxy when the data-api is unavailable.

## What the insiders did

**Accumulation, not a single bet.** None placed one large order. ricosuave666
made 32 buys over 17 hours on the Friday contract, clips of $100 to $2,800.
`0x1caa` made 29 buys over 56 hours, clips of $1,100 to $15,300, VWAP 0.11.
`0x4564` made 3 buys in its last 22 hours, $300 then $10,000 then $8,400.
The June 2026 screen's "big conviction single bet" archetype does not describe
any of them.

**Topical concentration.** Every market each wallet touched was the same
event family: Israel or US strikes on Iran, with one Khamenei leg. Lifetime
distinct markets were 5, 6, and 15. The crowd's heavy buyers in the same
windows had 400 to 1,100 lifetime markets.

**Buying stopped before the news.** Last winning-side buys landed 2.6 h,
2.5 h, and 0.9 h before the event. None bought after the news broke. Among
the crowd, a large share of pre-window volume arrives in the final hour as the
price starts to move.

**Date laddering and rolling.** ricosuave666 bought the "before July" contract
at 0.31 to 0.66, then sold it down and rolled into the cheaper "by Friday"
contract at 0.14 to 0.28 once that existed: same thesis, sharper date, higher
payout. `0x1caa` held the Feb 28 contract as its core and sprinkled $3,000 on
Mar 1 and small sums on Feb 26 and Feb 27 day-legs that expired worthless.
`0x4564` bought $300 each on Mar 2, 3, 4, and 5 the day before, a ladder
around an uncertain date. The losing legs are consistent with knowing the
operation was imminent but not the exact hour.

**Side flip.** `0x4564` is the most distinctive. From Feb 20 to Feb 25 it
bought NO on four day-legs at 0.87 to 0.97, the "nothing happens" trade. At
02:10 on Feb 27, 28 hours before the strike, it sold all NO positions and
started buying YES, finishing with $10,000 and $8,400 clips at 03:11 and
05:19 on Feb 28. A wallet that reverses its stated view and then goes heavy
on the opposite side a day before the event is a stronger tell than freshness
alone.

**Entry price looked like the market.** VWAPs were 0.11 to 0.22 on the
strike contracts and 0.49 on the broad "before July" contract, close to the
crowd's VWAP in the same windows. Entry price does not separate these wallets
from the crowd within a window. It only looks informative across windows,
which is the tautology the June screen fell into.

**Post-entry drift was zero or negative.** Mean drift over the 7 days after
entry, excluding the final 24 h before resolution, was −0.023, −0.013, and
−0.010. The markets did not move toward these wallets until the event itself.
The one honest signal from the June screen does not apply to military-strike
episodes, which are binary surprises. Drift should not be a scorer feature
for this class of event.

**After the event.** `0x4564` sold its last position at 09:53 on Feb 28 and
has no data-api activity since: burner behaviour. `0x1caa` kept trading Iran
markets through March across 16 markets. ricosuave666 went dormant for seven
months then returned for the January 2026 bets. Post-event behaviour varies
and is not available at scoring time anyway.

## Where they sat in the crowd

Winning-side buyers in the 72-hour window, and the published wallet's rank
under three naive orderings.

| Market | Buyers | Fresh (≤7 d) | Wallet | Rank by notional | Rank by notional among fresh | Rank by share × notional |
|---|---:|---:|---|---:|---:|---:|
| Israel action before July | 525 | 143 | `0x0afc` | 6 | 4 | 8 |
| Israel action by Friday | 211 | 67 | `0x0afc` | 1 | 1 | 1 |
| US strikes Iran by Feb 28 | 1,995 | 945 | `0x1caa` | 4 | 2 | 1 |
| US strikes Iran by Feb 28 | 1,995 | 945 | `0x4564` | 25 | 7 | 20 |
| US strikes Iran by Mar 1 | 1,222 | 535 | `0x1caa` | 45 | 12 | 135 |
| US strikes Iran by Mar 1 | 1,222 | 535 | `0x4564` | 300 | 131 | 841 |

Share means the wallet's window notional divided by its lifetime corpus
notional.

Reading: on the primary contract of each case, notional among fresh wallets
puts every published wallet in the top 7. Notional alone is beaten by
market-makers with 20,000 to 80,000 lifetime trades who are present in every
window; a freshness or lifetime-market-count filter removes them. On the
secondary contracts the published wallets were minor participants, so the
scorer should be judged on the case's primary market, with secondary markets
reported but not pooled.

## The crowd above and beside them

The wallets that out-ranked or sat next to the published ones among fresh
buyers are not noise. Several look like undisclosed members of the same
episodes.

| Wallet | Case | Window buys | Window $ | First activity | Lifetime markets | Note |
|---|---|---:|---:|---|---:|---|
| `0xd5de5cad` | Israel Jun 2025 | 4 | 25,002 | 2025-06-12 12:19 | 3 in corpus, 16 on api | #2 on Friday contract, 11.7 h before; appeared in the June screen's top-PnL list at $116K |
| `0xf1ff5353` | Israel Jun 2025 | 2 | 13,991 | 2025-06-12 12:47 | 5 | #4 fresh, opened 28 minutes after `0xd5de` |
| `0x296843af` | Israel Jun 2025 | 2 | 10,000 | 2025-06-12 17:43 | 1 | 2 trades ever, 6.3 h before, redeemed and gone |
| `0xc0a00575` | Israel Jun 2025 | 77 | 49,781 | 2025-06-11 12:50 | 5 | #1 by notional on "before July", 185 trades in 13 days then gone |
| `0xdde15ebd` | Feb 28 2026 | 78 | 29,826 | 2026-02-28 00:07 | 1 | 78 buys in 28 minutes starting 6.1 h before |
| `0x38745db2` | Feb 28 2026 | 5 | 26,514 | 2026-02-26 21:44 | 1 | single market; in the June screen's top list at $174K |
| `0xf5be0285` | Feb 28 2026 | 2 | 50,000 | 2026-02-27 21:52 | 1 | $50K in two clips 8.4 h before on the Mar 1 contract, then redeemed; in the June screen's top list at $97K |
| `0xa4eb5222` | Feb 28 2026 | 30 | 11,167 | 2026-02-28 00:55 | 2 | this is "nothingeverhappens911", whose address was truncated in the press; recovered here from the prefix |

The Israeli indictment says the civilian used four newly created accounts.
`0x0afc`, `0xd5de`, `0xf1ff`, and `0x296843af` were all opened June 10 to 12,
all bought YES on the Friday contract in the final 17 hours, and three of
four went quiet afterwards. That is a plausible reconstruction of the four,
and it came out of a plain ranking, not a search for them.

Random crowd wallets in the same windows, by contrast, were either long-lived
accounts with hundreds of markets placing $13 to $1,600, or brand-new accounts
placing under $100. Freshness alone is common: 27 % of the June window's
buyers and 47 % of the Feb 28 window's buyers were under 7 days old. Freshness
with size is rare.

## Anchor calibration

Hourly side-normalized VWAP in the 48 hours before each news timestamp:

- Israel "by Friday": 0.08 at −22 h, 0.28 at −12 h, 0.30 at −1 h, 0.98 at +0 h.
- Israel "before July": 0.27 at −32 h, 0.49 at −30 h, 0.57 at −2 h, 0.99 at +0 h.
- US strikes by Feb 28: 0.07 at −22 h, 0.24 at −17 h, 0.26 at −1 h, 0.87 at +0 h.
- US strikes by Mar 1: 0.11 at −22 h, 0.29 at −17 h, 0.30 at −1 h, 0.89 at +0 h.

In every market the price roughly tripled 17 to 30 hours before the event and
then held. An automatic anchor placed at the start of the largest 24-hour
move will land about a day before the news anchor. The published wallets'
first buys were at −55 h, −17 h, −57 h, and −22 h, so a window of 72 hours
before the auto anchor still contains them, but a window of 24 hours before
the auto anchor would miss two of three. The spec's 72-hour default is
adequate. The auto anchor is early, not late, which is the favourable
direction.

## Negative example

`0xefd06d1a` (Jan 2026, bets lost) had the same surface: fresh on Jan 7,
Iran-strike markets only, 17 markets. Differences: it traded both sides,
selling and re-buying across 14 sells in 25 trades on the Jan 31 contract,
and it spread across many day-legs over three weeks rather than concentrating
in the final three days. Its behaviour reads as an active speculator on the
same theme, not a positioned-and-waiting account. Two-sided activity and a
long spread of activity relative to the window are candidate negative
features, with one example only.

## What this means for the scorer

Features that separated the published wallets from the crowd in these windows:

1. **Freshness**: age at first window buy, from data-api first activity with
   corpus first trade as fallback. All three were ≤ 10 days; two ≤ 2.5 days.
2. **Window notional** on the primary market. All three were top 25 of
   thousands, and top 7 among fresh wallets.
3. **Lifetime breadth**: distinct markets ever traded. 5, 6, 15 versus
   hundreds for the market-makers who dominate raw notional.
4. **Concentration**: window notional as a share of lifetime notional. 0.26,
   0.54, 0.28 versus ≈ 0 for every high-notional non-fresh wallet.
5. **Side flip**: sold the opposing side then bought the winning side inside
   the window. One of three, but when present it is decisive.
6. **Timing shape**: last buy before the news, no post-news buying; buying
   spread over several hours rather than one clip. Weakly separating.

Features that did not separate, and should be excluded or left descriptive:

- Entry price within the window.
- Post-entry drift.
- Single-bet size or conviction fraction. The insiders clipped.

A deliberately simple composite, freshness-gated notional with a breadth
penalty and a flip bonus, would have ranked all three published wallets in
the top 5 of their primary market, and would have surfaced the four to eight
undisclosed candidates above as well. The holdout test will say whether that
generalises to the Maduro case, which has one published wallet with nine
corpus trades.
