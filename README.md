# Can a Defense Predict Run vs. Pass Before the Snap?

A play-calling analysis and scouting tool built for defensive game planning, using 34,502 run and pass plays from the 2025 NFL season to identify which pre-snap factors most influence whether an offense runs or passes, and to test whether a team's recent games can predict its next one.

NOTE: The target is qb_dropback (1 = dropback, including sacks and scrambles; 0 = designed run), so it measures the play that was called, not how the play ended. Every result is compared to nflfastR's own expected-pass model (xpass), which is used only as a benchmark and never as a feature.

📓 Full analysis notebook: [nfl_version_range.ipynb](nfl_version_range.ipynb)

## Business Problem

A defensive coordinator has to pick personnel, coverage, and blitz calls before the ball is snapped. This project identifies which situational factors (down, distance, field position, score differential, time remaining, timeouts, tempo, and offensive personnel) most strongly predict a run or a pass, and builds a scouting tool that trains on the weeks a coach could actually study before a game and then tests on that game.

## Summary of Findings

Our best league-wide model (logistic regression) predicts run vs. pass correctly 69.1% of the time on late-season games it had never seen (weeks 17–22, 5,478 plays), beating the simple strategy of guessing "pass" on every play (59.7%) by about 9 points.

However, nflfastR's xpass model scored 70.7% on the same plays. Our models beat the baseline, but they do not beat a professional league-wide model — and the same thing happened at the team level when we tested across a full season. For the Rams, training on the previous 3 games and testing on the next one, game by game across 17 games (1,083 plays), logistic regression averaged 66.4% vs. 70.1% for xpass and 60.0% for the baseline. For the 49ers, using an 8-game window across 11 games (661 plays), logistic regression averaged 67.3% vs. 71.4% for xpass and 56.9% for the baseline.

So far, scouting a single team's recent games does not predict that team better than a league model built on more data.

### What actually drives the call

Holding all else equal, down and distance matter most, with personnel close behind:

- Down and distance are the strongest signals. Later downs and longer yards-to-go push hard toward the pass — these were the two largest coefficients in the logistic regression.
- Wide receiver count is the strongest personnel signal. More receivers on the field means more passing, and it was the top feature in the decision tree (about 40% of its splitting power).
- Score and clock matter. Teams that are trailing pass more, and teams that are leading run more.
- Personnel only makes sense as a swap. With 11 players on the field, adding a player always means taking one off. For the Rams (weeks 1–3), swapping a receiver for a tight end made a pass about one-third as likely (odds ratio 0.33). For the 49ers (weeks 1–8), swapping a running back for a receiver made a pass about 1.3 times as likely (odds ratio 1.34).

Note: team-level effects come from only a few hundred plays, so these numbers should be read as directionally correct but imprecise.

## Where the model struggles

Runs are much harder to spot than passes. League-wide, the model correctly catches 80% of passes but only 52% of runs.

One game is too small to judge a model. A single game is about 65 plays, which gives a margin of error of roughly ±11 points. When we trained the Rams on weeks 1–3 and tested on week 4, logistic regression beat xpass (73.1% vs. 71.6%), but over the full season it lost (66.4% vs. 70.1%). That one-game win was luck.

Recent tendencies can flip. When we trained the Rams on weeks 5–7 and tested on week 9 (week 8 was a bye), guessing their recent majority play scored only 46.6% — worse than a coin flip — because they called the opposite of their recent tendency that game.

There is too little data to tune on short ranges. With a 3-game training range (about 175 plays), GridSearchCV's cross-validation accuracy was sometimes below the baseline, which means the "best" settings it picked were mostly noise.

## Recommendations

Use the baseline charts to show tendencies, and the models to make predictions. They answer two different questions — "what does this team usually do?" and "what will they call on this play?" — and a coaching staff needs both.

- Use the baseline charts as the scouting report. The run/pass percentage charts by down and distance (Step 3), by personnel (Step 4), and by down and distance and personnel combined (Steps 5 and 5b) show a team's tendencies in a form a coach can read at a glance. For example, a coach can see how often a team passes on 3rd & 4–6 out of 1 RB, 1 TE, 3 WR, without needing to understand a coefficient or an odds ratio.
- Start with the combined chart. Down and distance and personnel together are where tendencies are sharpest, because the same personnel group can be a run look on 1st & 10 and a pass look on 3rd & 7–9. The single-factor charts are useful for the big picture, but the combined heatmap and stacked bars are where the scouting value is.
- Always read the play count (n) on each bar. A 100% pass rate on 4 plays is not a tendency — it's 4 plays. The charts label every bar with its count, and the Step 5b chart hides combinations with fewer than 5 plays (MIN_PLAYS) for this reason. Treat anything under about 10 plays as a hint, not a rule.
- The charts hold up about as well as the models. When we trained the Rams on weeks 1–3, simply guessing each personnel group's majority play scored 73.1% on week 4 — the same as logistic regression and better than xpass (71.6%). For one game, the tendencies shown in the charts captured about as much as the models did.
- Use the models for play-by-play predictions, and use xpass as the starting point. Over a full season, xpass was the most accurate option in every test we ran, so a prediction tool should start there rather than with a team-only model.
- Keep the charts recent and check them each week. Tendencies can flip — the Rams' recent majority play scored only 46.6% in week 9 — so the charts should be rebuilt from the most recent games before each opponent, which is what the week range input in Step 1 does.

## How to Use the Scouting Tool

The notebook includes an interactive tool for scouting one team. Open the notebook and run all cells. In Step 1 of the Individual Team Analysis section, you'll be prompted for a team abbreviation (e.g., LA, SF, KC — the Rams are LA, the Chargers LAC, and the Raiders LV) and a range of training weeks (e.g., 1-3).

The notebook trains on those weeks and tests on the team's next game. If that week was a bye, it automatically uses the next game played and tells you. All scouting charts use only the training weeks, so they never peek at the game being predicted.

To skip the prompts (for example, to use Restart & Run All), set these at the top of the Step 1 cell:

```python
TEAM = "LA"          # team abbreviation
WEEK_RANGE = "1-3"   # train on weeks 1-3, test on the next game
```

## Methodology (Technical Summary)

- Data: 2025 NFL season from nflreadpy — load_pbp() for situational context (down, distance, field position, score differential, time remaining, timeouts, home/away, division game, no-huddle) and load_participation() for offense_personnel, joined on game_id and play_id
- Cleaning: kept only run and pass plays (removing punts, kicks, kneels, spikes, and no-plays) and dropped plays with no down (two-point conversions); no duplicate plays were found
- Feature engineering: personnel parsed into counts, with fullbacks counted as backs (they appear on 8.6% of plays) and offensive linemen counted separately (6 or more linemen appear on about 5.5% of plays), plus a home/away flag, down-and-distance buckets, and personnel group labels
- Models tested: Logistic Regression, Decision Tree, and KNN, each tuned with GridSearchCV
- Baselines: always guess the majority play; the majority play within each down-and-distance or personnel group; and nflfastR's xpass (guess pass when xpass ≥ 0.5)
- Validation: data is split by week so plays from the same game never appear in both training and test sets, cross-validation uses GroupKFold by week for the same reason, and a rolling check repeats "train on the previous N games, test on the next" across every game of the season
- Personnel interpretation: n_rb is left out of the models as the reference position, so each personnel coefficient reads as "swap one running back for this position"; when a team's running back count never changes, the notebook reports the swap the team actually made instead
- Evaluation metric: accuracy, chosen because the classes are only mildly imbalanced (about 60% pass, 40% run) and a defensive coordinator cares most about how often the pre-snap read is right; accuracy is reported with a 95% margin of error, and the classification report is used to check run and pass recall separately

See the full notebook for detailed exploratory analysis, visualizations, model comparisons, and feature importance details.
