# Closed-loop simulation recordings

All videos are generated from the released best checkpoint policies. The solid robot is controlled by the Actor in MuJoCo Warp; the translucent blue ghost is the stored reference. These are simulations, not real-robot tests. Source trajectories and evaluation results are separate from a visual preview.

## LAFAN1 best

[Animated preview](lafan1.gif) · [Combined MP4](lafan1.mp4)

| Reference | Recording | Scope |
| --- | --- | --- |
| Dance | [dance1_subject1.mp4](lafan1/dance1_subject1.mp4) | First 12 seconds |
| Running | [run1_subject2.mp4](lafan1/run1_subject2.mp4) | First 12 seconds |
| Jumps | [jumps1_subject1.mp4](lafan1/jumps1_subject1.mp4) | First 12 seconds |

These excerpts are not whole-clip success tests. The independent saved [40-clip report](../results/lafan1_best.json) supplies the quantitative result.

## Six-motion specialist

[Shanshenbu GIF](shanshenbu.gif) · [All six complete recordings](shanshenbu.mp4)

| Motion | Complete MP4 |
| --- | --- |
| 闪身步 / Shanshenbu | [shanshenbu.mp4](skills/shanshenbu.mp4) |
| Pro turning and jumping | [shanshenbu_pro.mp4](skills/shanshenbu_pro.mp4) |
| Four-motion sequence | [shanshenbu_combo4.mp4](skills/shanshenbu_combo4.mp4) |
| 狗熊躲猫猫 | [gouxiong_duosuomao.mp4](skills/gouxiong_duosuomao.mp4) |
| 浪子踢球 | [langzi_tiqiu.mp4](skills/langzi_tiqiu.mp4) |
| 摇头晃脑 | [yaotou_huangnao.mp4](skills/yaotou_huangnao.mp4) |

## Recording settings

- Deterministic clip starts, seed 0, observation/action/reset noise disabled, domain randomization disabled.
- Physics control at 50 Hz, rendered at 25 fps and 960×540; GIFs use 10 fps and width 560.
- Per-clip recording stops at failure, clip end or an explicitly requested demonstration cap. Failed or reset states are not substituted with reference animation.
- Reference and checkpoint hashes, recorded frame counts and end/failure flags are retained in [recordings.json](recordings.json). The specialist records all six clips to their end; LAFAN clips are capped excerpts.

Regenerate with the `record` commands in the [main README](../README.md#save-videos-and-gifs). Motion attribution and media rights remain as described in [third-party notices](../THIRD_PARTY_NOTICES.md#motion-data-and-demonstration-media).
