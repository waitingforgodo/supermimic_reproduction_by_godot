# Third-party notices

Godot's training recipes, integration, validation, packaging and trained results are maintained in this repository. Existing copyright and license notices are retained for the components below. This project is a community reproduction and adaptation, not an official implementation by the SuperMimic paper authors.

| Component | Relationship | License / notice |
| --- | --- | --- |
| [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), commit `ba6105349069bf040f1fdad60873d13138d505a2` | Mathematical training core, environment, FK, data conversion and benchmark adapted under `src/g1_supermimic/core/`; rendering adapted separately. Modules have been relocated and integrated with Godot's training extensions. | MIT, Copyright 2026 Yuxi Liu. Full notice: [licenses/supermimic-core-MIT.txt](licenses/supermimic-core-MIT.txt). |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab), commit `77183062eb92026bfbee2daf276f87cad36fd957` | Reference for real-observation normalization, dynamics normalization and relative anchor orientation. Local adaptation retains this project's observation/state conventions. | See the reference project; attribution is not a claim of identical implementation. |
| [NJU SuperMimic reproduction](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real), commit `f8f60d7bbdba68468c050ef100af02d752ca238e` | Source of six custom robot-motion references and reference for return-normalized sampling / visit-count initialization. Its separate trainer and released policies are not included. | MIT, Yuxi Liu and Hongyu Ding. Full notice: [licenses/custom-motions-MIT.txt](licenses/custom-motions-MIT.txt). Its [motion/data notices](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/blob/main/LICENSES/THIRD_PARTY_NOTICES.md) also apply to the source material. |
| [TrackerLab command/style experiment](https://github.com/Renforce-Dynamics/trackerlab-supermimic/blob/3b8e31ea36183c4ccc1f5c3a023bfa292d141692/docs/supermimic_command_style.md) | Method reference for optional LS discriminator style shaping through imagined dynamics. This repository keeps reference-conditioned tracking. | The local adapter is independently integrated; the reference project is MIT. |
| [xMimic](https://github.com/nico-bohlinger/xmimic) | Method reference for EMA / training-efficiency experiments. Its source, assets and policies are not bundled. | Consult the reference repository for its own code license. |
| [mjlab](https://github.com/mujocolab/mjlab), MuJoCo, MuJoCo Warp, NVIDIA Warp | Installed simulation and model dependencies. No robot meshes/MJCF are copied into this Git repository. | Their package licenses and robot-asset notices remain applicable. |
| PyTorch, NumPy, SciPy, Viser, Pillow, imageio-ffmpeg and other installed dependencies | Runtime, visualization and utilities. | Their respective licenses; dependencies are installed separately. |

## Motion data and demonstration media

- **LAFAN1 is not redistributed.** Obtain it from the [Ubisoft La Forge dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and the [G1 retargeting release](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). The dataset is identified by its publisher as **CC BY-NC-ND 4.0**; downloading and using it requires observing those terms. `data/lafan1_source.json` pins the CSV revision and object hashes, without containing motion samples.
- `data/skills/*.npz` contains converted robot trajectories sourced from the MIT custom-motion release above. The original video creators retain rights to their videos; source videos, human body models, reconstruction networks and their weights are not included.
- `media/` shows trained policies running in simulation, with a blue reference ghost. These illustrations and their underlying motion content are not relicensed as MIT by the code license. Keep source attribution and observe the applicable data/media terms.
- Released checkpoints are Godot's local training results. Model cards identify their training data; distributing the weights does not grant additional rights to the underlying datasets or robot assets.

The code license does not override any third-party data, media, model or asset license.
