# End-game experiment — Step 0: headroom

- ThresholdAgent aggression (frozen): **0.2** (calibration tried: 0.2→43.1%, 0.0→43.6%, 0.5→44.0%, 1.0→46.9%, 2.0→50.2%; in 55–85% window: False)
- ppo_best win rate vs ThresholdAgent (5000 games): 42.8%
- Games reaching a close end-game (`f_close`): 54.3%
- Games that reach a close end-game and ppo_best loses: 29.8% (52.2% of its losses)
- Headroom bound (×0.3): **8.9%** of full-game win rate
- MDE full-game equivalent (3 pp × f_close): 1.6%; gate requires bound ≥ 2× that
- Score-awareness (draw rate trailing − leading, matched states): 37.1% (weight 10381) — informational
- Real-position suite: 1998 positions, per band {'150-174': 666, '175-189': 666, '190-199': 666}
- Burn distribution: 861 samples, deck-size quartiles [0, 17, 23, 29, 73]

**Go/no-go: GO**
