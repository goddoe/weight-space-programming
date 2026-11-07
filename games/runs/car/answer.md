The policy has learned the correct direction: action **1** turns the wheels counter-clockwise, driving left toward blue. It also gets onto the ramp. The failure is its **drive/coast switching**, not reversed controls.

The weights support that diagnosis: hidden unit 116 strongly receives horizontal velocity (`embed.W[54,116] = 2.25`, `[236,116] = 2.37`, `[254,116] = 1.73`). The output head strongly opposes coasting and forward drive—for example, row 6 contributes `+0.683` to off and `−0.638` to forward; row 114 contributes `+0.667` and `−0.641`. In the trace, this switching persists while position stops changing. Motor-off supplies no holding torque, so intermittent propulsion is insufficient to maintain the uphill climb. The policy appears trapped at a partial-progress solution.

```edits
actor_head.b[0] = -1000
```

This removes the coasting option from the sole active motor binding, preventing repeated loss of propulsion on the ramp. Every hidden-layer weight and both powered-action logits remain unchanged: the trained policy still chooses between forward and reverse with exactly the same relative odds.

The intended result is a sustained climb and enough launch speed to clear red. This is an untested intervention, not a verified success-rate claim.