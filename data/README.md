# Reference motions

`skills/` contains six 50 Hz G1 trajectories used by the specialist checkpoint:

| File | Motion |
| --- | --- |
| `shanshenbu.npz` | 闪身步 / Shanshenbu |
| `shanshenbu_pro.npz` | Pro turning and jumping variation |
| `shanshenbu_combo4.npz` | Four-motion continuous reference |
| `gouxiong_duosuomao.npz` | 狗熊躲猫猫 |
| `langzi_tiqiu.npz` | 浪子踢球 |
| `yaotou_huangnao.npz` | 摇头晃脑 |

The robot-motion source is the MIT release identified in [third-party notices](../THIRD_PARTY_NOTICES.md). Only converter metadata was made portable; model-consumed numeric arrays are unchanged.

LAFAN1 samples are downloaded separately:

```bash
bash g1.sh prepare
```

This fetches exactly the 40 G1 CSV clips pinned by `lafan1_source.json`, converts them from 30 Hz to 50 Hz using CPU kinematics, and writes ignored `data/lafan1/` files. Existing valid downloads are reused. See [dataset terms](../THIRD_PARTY_NOTICES.md#motion-data-and-demonstration-media).

The checkpoint contract hashes the actual motion arrays, names and frequency. ZIP timestamps and converter source-path metadata do not affect it. A numerical change to a motion is rejected by normal resume. Use `--init-from` to start a new specialist task, or explicit `--allow-new-motion` for unseen-motion replay/evaluation.
