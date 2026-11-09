# Best trained models

Full checkpoints, including world model, Actor, Critic and their optimizers, are attached to the repository's [v0.1.0 Release](https://github.com/waitingforgodo/supermimic_reproduction_by_godot/releases/tag/v0.1.0). They exceed GitHub's ordinary 100 MiB per-file limit, so Git stores their cards, evaluation reports and checksums; Release assets store the weights.

```bash
bash g1.sh models
# Or download just one:
bash g1.sh models --only skills
```

The downloader verifies both the archive and checkpoint SHA-256 before installing `models/lafan1/best.pt` or `models/skills/best.pt`. It preserves an existing different checkpoint. Only load checkpoint files you trust.

| Model | Best control-step counter | Training clips | Whole-clip success | Mean full-clip normalized return |
| --- | ---: | ---: | ---: | ---: |
| LAFAN1 | 125000 | 40 | 40/40 | 0.847032 |
| Six-motion specialist | 2000 new steps | 6 | 6/6 | 0.838217 |

Scores are on each model's own training set; they do not establish generalization and should not be compared across the two datasets. The specialist starts from the LAFAN1 model. Published metadata uses portable paths and content hashes, while every model and optimizer tensor remains unchanged. See [export verification](../results/model_export_validation.json).

The published best models use FP32 without EMA or AMP auxiliary rewards. Those options are experiments, not the source of the above results. Training data and asset rights remain as described in [third-party notices](../THIRD_PARTY_NOTICES.md).
