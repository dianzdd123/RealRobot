# Reusing RealRobot

Occlusion-aware, language-guided robotic grasping.

**Current scope:** Hardware research pipeline using xArm, RGB-D perception and external language/model services.

This repository is a fork of [cb20600/RealRobot](https://github.com/cb20600/RealRobot). Original authorship, citations and upstream notices are retained. Changes here apply to this fork.

## Setup and inspection

Create an isolated Python environment and review `requirements.txt` before installing. It includes hardware SDKs and a SAM2 source dependency; model weights and physical devices are not included by package installation.

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python scripts/check_repository.py
```

These are dependency and structural checks only. The installation has not been verified against every CUDA, camera-driver or operating-system combination.

## Pipeline map

| Stage | Code | Inputs / outputs |
|---|---|---|
| Orchestration | `robot.py`, `robot_with_occ.py` | Task and camera observation → grasping workflow |
| Detection and segmentation | `utils/yolo_utils.py`, `utils/sam2_utils.py` | RGB-D frame and model weights → object regions |
| Geometry | `utils/coordinate.py`, `utils/camera_to_world.py`, `utils/rotate.py` | Depth, intrinsics and extrinsics → world-space grasp estimates |
| Reasoning | `utils/API.py`, `utils/occluded.py` | Scene descriptions and instruction → target/occlusion decisions |
| Motion | `utils/traj_execute.py` | Proposed grasp/trajectory → physical robot commands |
| Reference observations | `test_imgs/`, `imgs/` | Recorded PNG/NPZ observations and visualisations |

## Configure before connecting hardware

Review robot addresses, camera serial/settings, local checkpoint/config paths, API configuration, output folders and every calibration transform. Included calibration numbers describe the original equipment and must be replaced for another installation. Keep credentials in local environment configuration, outside Git.

Start with recorded-image perception, then inspect world-coordinate estimates and proposed motion independently. `robot.py` and `robot_with_occ.py` can communicate with real equipment and services; running them is not an installation test. Confirm workspace, units, frame conventions and a stop procedure on the actual system.

## Reuse and troubleshooting

Use small geometry/perception functions independently where possible. Preserve the matching RGB/depth pair and camera intrinsics; a depth visualisation PNG is not a replacement for metric depth. Confirm array keys and units in NPZ observations before loading your own recordings.

Missing weights/configuration produce model-loading failures; connection errors point to the robot/camera/service setup; implausible world positions point first to depth scale or calibration. Keep raw observations separate from generated overlays and logs so repeated experiments do not overwrite inputs.

No hardware execution, external model download, paid API call or grasp-success reproduction is included in the structural validation.

## Attribution and reuse permissions

No repository-wide licence file is included in this checkout. Confirm reuse and redistribution permissions with the relevant rights holders; this maintenance update does not add a licence or relicense upstream code, data or assets.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the local workflow and what to include with a proposed change.
