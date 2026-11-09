# Measured performance and evaluation protocol

All headline quality values describe Godot's own saved local training results. They are not paper/author benchmark numbers. Reports retain per-clip scores rather than only a gallery or aggregate.

## Tracking quality

The deterministic whole-clip protocol starts each clip at its first reference frame, disables action/observation/reset noise and domain randomization, and runs until completion or failure. A successful full clip is required; surviving a short excerpt is insufficient. Failed clips receive zero reward for their remaining frames when computing full-clip normalized return.

Candidate selection ranks whole-clip success rate, mean completion and then mean full-clip normalized return. A prospective improvement is confirmed twice and the worse lexicographic score is kept. The LAFAN1 best completes 40/40 clips at 0.847032. Six-motion fine-tuning completes 6/6 at 0.838217, versus its own zero-step target-set baseline of 0.754944. Both are training-set measurements, not held-out generalization.

The LAFAN run's final recorded stage added 20000 control steps in 45095.9 seconds (12.53 hours), ending at counter 126000 and selecting counter 125000. That duration is **only that stage**, not the complete historical cost from initialization. The specialist added 2000 steps / 784 optimization updates in 3821.5 seconds. Sources: [LAFAN resources](../results/lafan1_resources.json), [specialist resources](../results/skills_resources.json).

## Same-budget update-time comparison

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GiB. Each case starts from the same 125000-step parent, uses 4096 simulation environments, H32, policy batch 256, world-model batch 512 × 10 and the same learning-rate schedule. Each adds 64 control steps and 10 optimizer updates.

| Measurement | FP32 reference | BF16 cached weights + EMA |
| --- | ---: | ---: |
| World-model update median | 1.046 s | 0.883 s |
| Policy + Critic update median | 3.574 s | 3.328 s |
| Paired update median | 4.620 s | 4.175 s |
| Training segment before final evaluation | 55.819 s | 52.394 s |
| Peak PyTorch allocated memory | 4.614 GiB | 4.420 GiB |
| Whole clips completed | 40/40 | 40/40 |
| Full-clip normalized return | 0.847682 | 0.846953 |

The paired-update timer sums corresponding WM and policy/Critic timings, excludes the first update, then takes the median of the remaining nine pairs. It includes update internals, gradient hooks and EMA, and excludes collection, post-update state checks, saving and evaluation. The training-segment timer includes replay warm-up and saving but excludes simulator/network construction, capacity probing and evaluation.

Observed reductions: **9.64%** paired update time, **6.14%** measured training segment and **4.21%** peak allocated memory. These are a single bounded comparison, not a long-run convergence result or a claim about overall training throughput. BF16 did not improve score; the released best models remain FP32 without EMA. [Raw comparison](../results/performance_comparison.json).

## Reproducing checks

Use the pinned environment and preserve environment count, H32, batches and parent checkpoint when comparing variants. Treat `--iterations` as total control steps. A short timing run should use `--stop-after-steps` so its learning-rate schedule stays fixed. Do not compare separate runs while changing other GPU load.

`scripts/benchmark_actor.py` measures only Actor inference latency, with warm-up excluded and device synchronization when applicable. It uses synthetic tensors and reports that fact; it does not measure sensor processing, physics, DDS communication, actuator latency or a complete hardware control deadline.

The released specialist Actor was measured on CPU with two PyTorch threads, batch size 1, 20 warm-up calls and 200 timed calls: **2.495 ms median, 3.100 ms p95**. This was a shared-workstation measurement with other host activity, not an isolated CPU/GPU comparison. [Raw Actor timing](../results/actor_latency_cpu.json).

```bash
python scripts/benchmark_actor.py --checkpoint models/skills/best.pt \
  --device cpu --threads 2 --samples 200 --output outputs/actor_latency.json
```
