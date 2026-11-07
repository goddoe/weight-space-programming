The failure is a stalled approach, not reversed action meanings. Prominent head rows, such as row 6 (`[0.683, -0.638, -0.005]`), strongly distinguish coasting from leftward drive. The rollout shows this switch repeatedly operating without producing further movement; reverse is effectively never sampled.

The car climbs the ramp but loses its momentum with the green wheel suspended above red. That earns approximately `0.2 × 0.76 = 0.152` shaping reward without risking the terminal penalty. Simply continuing slowly would risk dropping onto red.

I retain the bulk of the trained network and add a small, isolated run-up controller: settle, back up at limited speed, turn, and accelerate across the ramp. The remaining trained controller resumes after the rear wheel clears the crest.

```edits
# Observation channels for the corrective branch.
embed.W[:, 6:9] = 0
embed.b[6:9] = [0, 10, 0]
embed.W[52, 6] = 1
embed.W[54, 7] = 1
embed.W[252, 8] = 1
embed.W[:, 15] = 0
embed.b[15] = 0
embed.W[253, 15] = 1

# Detect rear-wheel height, reverse speed, turning position,
# horizontal direction, and whether the rear wheel has cleared the ramp.
actor0.W[6:9, :] = 0
actor0.W[15, :] = 0
actor0.W[:, 123:128] = 0
actor0.b[123:128] = [-68, -228, -395, -295.5, -196]
actor0.W[15, 123] = 100
actor0.W[7, 124] = 20
actor0.W[6, 125] = 100
actor0.W[7, 126] = 30
actor0.W[8, 127] = 100

# Reverse/coast phase versus leftward launch phase.
actor1.W[123:128, :] = 0
actor1.W[:, 125:128] = 0
actor1.b[125:128] = [10, -10, 0]
actor1.W[123, 125] = 10
actor1.W[124, 125] = 10
actor1.W[125, 126] = -10
actor1.W[126, 126] = 10
actor1.W[127, 127] = 5

# Relay the branch without feeding it into the retained controller.
actor2.W[125:128, :] = 0
actor2.W[:, 125:128] = 0
actor2.b[125:128] = 0
actor2.W[125, 125] = 5
actor2.W[126, 126] = 5
actor2.W[127, 127] = 5

# Separate coast, reverse, and forward commands.
actor3.W[125:128, :] = 0
actor3.W[:, 125:128] = 0
actor3.b[125:128] = [-20, -20, -10]
actor3.W[125, 125] = 10
actor3.W[126, 125] = 10
actor3.W[127, 125] = 10
actor3.W[125, 126] = -10
actor3.W[126, 126] = 10
actor3.W[127, 126] = 10
actor3.W[126, 127] = -10
actor3.W[127, 127] = 10

# Convert command gates to approximately zero or one.
actor4.W[125:128, :] = 0
actor4.W[:, 58] = 0
actor4.W[:, 89] = 0
actor4.W[:, 90] = 0
actor4.b[58] = 3
actor4.b[89] = 3
actor4.b[90] = 3
actor4.W[126, 58] = 3
actor4.W[127, 89] = 3
actor4.W[125, 90] = 3

# Override only the physically connected action entry.
actor_head.W[58, 0:3] = [-100, -100, 100]
actor_head.W[89, 0:3] = [-100, 100, -100]
actor_head.W[90, 0:3] = [100, -100, -100]
```

The branch initially coasts while the rear wheel settles. It then backs toward body x ≈ 3.95, coasting whenever rightward speed exceeds approximately 1.4 or the rear wheel lifts.

At the turning point it drives left. Negative velocity keeps that command active during the run-up and climb, rather than returning to the original stall. Once rear-wheel x falls below approximately 1.96, the corrective logits disappear.

Large logit margins reduce unintended sampled actions during the maneuver. These are physics-based, untested edits—not a verified success rate.