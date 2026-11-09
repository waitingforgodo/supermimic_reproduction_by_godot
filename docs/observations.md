# Actor observation and deployment contract

The released Actor interface is identical for both models.

| Proprioception, once per control step | Width | Hardware source |
| --- | ---: | --- |
| Joint position minus training default pose | 29 | Joint encoder position in the model's joint order |
| Joint velocity minus its zero default | 29 | Joint velocity feedback |
| Pelvis angular velocity, pelvis/body axes | 3 | Calibrated pelvis IMU gyroscope |
| Gravity projected into pelvis axes | 3 | IMU attitude estimate; rotate world gravity into body axes |
| Previous clean normalized policy action | 29 | Saved by the controller, before position scaling/offset |

Each frame has 93 values. Ten frames are flattened **term-major, oldest to newest**: all ten joint-position frames, then all ten velocity frames, etc. On initialization, the first valid sample fills every history slot. The control rate is 50 Hz (0.005 s physics × decimation 4 in simulation).

The environment goal has 689 values: ten 68-value reference frames, followed by a 9-value reference-anchor pose. The Actor removes the last block's three positional values, leaving **686 effective goal values**. Reference features are joint offsets (29), joint velocities (29), projected gravity (3), body angular velocity (3), heading-frame linear velocity (3) and height (1). They come from the local motion file, not a forecast of the robot's actual future sensor state.

The remaining 6-value anchor orientation is the first two columns of `R_world_torso_robot.T @ R_world_torso_reference`, flattened in row order. On hardware, reconstruct robot torso orientation from pelvis IMU attitude plus measured waist yaw/roll/pitch through the training kinematic model. Align the initial yaw/reference world frame and account for heading drift. Robot global position and measured base linear velocity are not Actor inputs.

Fixed input scales and saved RunningNorm buffers are inside the Actor; deployment must preserve them exactly once and freeze their statistics. Joint velocities are divided by 5 and angular velocity by 2 before learned normalization. Do not substitute another controller's scale/default-angle arrays. Actions are normalized 29-value outputs; physical joint targets use the model-specific action scale and default pose.

The expected hardware is a compatible full 29-DoF G1 with the required waist/wrist joints and low-level access. Check the actual hardware model, PR joint convention, IMU axis calibration, encoder zeros, sample freshness, timing and motor-name mapping. An array slot alone does not prove a physical joint is available. See [Unitree's G1 joint definition](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py) and [state handling example](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/deploy_real.py).

The optional AMP discriminator and world model are training components and add no hardware Actor observations. No DDS command publisher or validated hardware controller is included. `deploy` runs simulation playback; completed simulation evaluation is not hardware validation.
