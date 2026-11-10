# G1D Sinopec test-tube task — semi-interactive auto-collect Quick Start

> Branch: `feature/g1d-sinopec-auto-collect`  
> **Do not merge into `main` until reviewed.**  
> 中文: [quick_start.md](./quick_start.md)

This flow collects ACT training demos for **right-arm pick → bimanual handover → left-arm place** (pose mode + gripper `target_percent`).

---

## 1. Repos / branches

| Repo | Checkout | Role |
|------|----------|------|
| `lerobot_ros2` | `feature/g1d-sinopec-auto-collect` | `auto_collect_g1d_tube.sh`, task YAML, hole meta |
| `data-collector-platform` | `feature/g1d-sinopec-auto-collect` | Gripper `target_percent` + G1D joint/pose config |
| `FaSim-Isaac` | `feature/g1d-sinopec-auto-collect` (optional) | Longer ros2_control activate waits on cold Play |

```bash
git -C ~/lerobot_ros2 fetch origin && git -C ~/lerobot_ros2 checkout feature/g1d-sinopec-auto-collect
git -C ~/data-collector-platform fetch origin && git -C ~/data-collector-platform checkout feature/g1d-sinopec-auto-collect
# optional
git -C ~/FaSim-Isaac fetch origin && git -C ~/FaSim-Isaac checkout feature/g1d-sinopec-auto-collect
```

---

## 2. Environment

Match your team’s FaSim / ros2_control setup, for example:

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=72          # change if your lab uses another id
export ROS_USE_SIM_TIME=true
```

Default data roots (override via env if needed):

- Raw: `/data/agent_client_installer/Agent/raw_data`
- Converted / LeRobot: `/data/agent_client_installer/Agent/data_collected`

---

## 3. Startup order (each session)

### 3.1 Isaac Sim

1. Open the Sinopec G1D `room.usda` (FaSim project path per your team).
2. **First Play** often spams PhysX “Simulation view object is invalidated”. When it settles: **Stop → wait ~5s → Play again**.  
   Second Play should show controllers active (e.g. `Successfully switched controllers` / HOLD).  
   **Do not Stop while the log says `ignoring Stop during activate`.**
3. Confirm topics: `/robot_description`, `/ocs2_wbc_controller/current_state`, cameras, grippers.

### 3.2 Collector

```bash
cd ~/data-collector-platform/unified_data_platform
# activate conda / source ROS as usual
python unified_collector_converter.py
```

Wait for `Node initialized`. The collect script starts/stops recording via `/teleop/controller_state` (`50` / `51`).

### 3.3 Auto-collect script

```bash
cd ~/lerobot_ros2/examples/IsaacSim
bash scripts/auto_collect_g1d_tube.sh
# optional default for the episode-count prompt:
bash scripts/auto_collect_g1d_tube.sh 20
```

---

## 4. Interactive flow

1. **How many episodes?** — planned count (you can still quit early with `q`).
2. Per episode:
   - **start hole 1–6**: pick from right rack `test_tube_rack_01` (default 6)
   - **place hole 1–6**: place into left rack `test_tube_rack` (default 3) → scene `place_hole_N`
3. Script automatically:
   - Re-enable both arms → FSM **HOME** → open grippers  
   - Teleport the tube to the start hole  
   - Publish `50` → `motion-generation … --no-reset` → publish `51`
4. **Keep this episode? [y/N]** — visual QA in Isaac (pick / handover / into target hole).  
   Soft CSV checks that left/right gripper CSVs contain `0` and `1` (you may still keep a soft-fail episode, but prefer discard).
5. **Continue collecting? [y / q]** — next episode or stop.

By default the script **does not** Stop/Play Isaac. Use `--with-sim-reset` only if you need a full sim reset (slow; wait for controllers again yourself).

---

## 5. On-disk data & conversion

- Each raw episode: `raw_data/YYYYMMDD_HHMMSS_xxxxxx/`
- The collector queues conversion into LeRobot under `data_collected/`
- Keep/Discard does **not** require waiting for conversion

Discard deletes the matching directories under both `raw_data` and `data_collected`.

---

## 6. Troubleshooting

| Symptom | What to do |
|---------|------------|
| Wait stack shows `ocs2_wbc=False` but topic exists | QoS: `current_state` is TRANSIENT_LOCAL; this script already uses latched QoS. Confirm second Play activated controllers. |
| Cold-Play PhysX spam | Stop → wait → Play; never Stop mid-activate. |
| Arms left mid-air after an episode | Script HOMEs before each take; if HOME fails, check `/fsm_command` and arm modes. |
| Tube misses the hole | Discard; tune insert params in `pick_tube_full.yaml` later. |
| motion-generation fails on `robot_description` / BASE_LOCK | Stack not ready or activate was interrupted; redo §3.1. |
| Gripper CSV always `1` or missing | Collector must be on this feature branch and subscribe to `/*/target_percent`. |

---

## 7. Scene conventions

- Right rack (pick): `/World/room/scene/test_tube_rack_01`, holes `hole_1`…`hole_6`
- Left rack (place): `/World/room/scene/test_tube_rack`
- Tube: `/World/room/scene/test_tube`
- Task: `pick_tube_full`, scene: `place_hole_N`
- Full-run `BASE_LOCK` (base fixed) so ACT EE trajectories stay reproducible

---

## 8. Minimal cheat sheet

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=72 ROS_USE_SIM_TIME=true

# Terminal A: Isaac Playing, controllers OK
# Terminal B:
cd ~/data-collector-platform/unified_data_platform && python unified_collector_converter.py

# Terminal C:
cd ~/lerobot_ros2/examples/IsaacSim && bash scripts/auto_collect_g1d_tube.sh
```

When reporting issues, attach `/tmp/g1d_auto_collect/run_*.log` and the Isaac activate / HOLD snippet.
