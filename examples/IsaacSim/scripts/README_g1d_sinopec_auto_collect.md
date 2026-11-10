# G1D Sinopec auto-collect (feature branch)

Branch: `feature/g1d-sinopec-auto-collect` — **do not merge to `main` until reviewed.**

## Prerequisites
- Isaac Sim Playing `room.usda` (FaSim G1D Sinopec env), ROS domain + zenoh as usual
- `data-collector-platform` on the same feature branch (gripper `target_percent` support)
- Optional: FaSim `feature/g1d-sinopec-auto-collect` for longer cold-Play activate waits

## Run
```bash
cd ~/lerobot_ros2/examples/IsaacSim
bash scripts/auto_collect_g1d_tube.sh
```
Prompts for episode count, start/place holes, keep/discard, continue/quit.
Between episodes: FSM HOME + tube teleport (no Stop/Play by default).
