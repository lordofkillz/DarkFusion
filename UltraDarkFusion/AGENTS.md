# DarkFusion Project Memory

## Automatic labeling for gameplay footage

- Treat generic missing-person, missing-enemy, and similar model-generated labels as unreliable for game footage.
- HUD and scene elements often resemble valid objects. Common false positives include flags, revive or respawn indicators, character faces and portraits, nameplates, icons, mountains, shadows, and other player-shaped scenery.
- Do not assume an unlabeled model detection is a missing annotation.
- Any future gameplay auto-labeling design needs strong game-aware HUD and distractor filtering, conservative thresholds, and validation on real footage before it can write dataset labels.
- Do not restore the removed missing-label suggestion workflow unless the user explicitly requests a new approach.
