#!/usr/bin/env bash
# Semi-interactive G1D pick→handover→place collection.
#
# Prerequisites: Isaac Playing (room.usda), collector running, ROS domain 72 / zenoh.
#
# At start you choose how many episodes. Per episode:
#   - start hole on the RIGHT rack (test_tube_rack_01)
#   - place hole on the LEFT rack  (test_tube_rack)
# After motion: keep/discard, then continue (y) or quit (q).
# Between takes: FSM HOME (arms) + tube teleport — no Isaac STOP/PLAY by default.
#
# IMPORTANT: default does NOT call Isaac STOP/PLAY. FaSim ROS2ControlLaunch ignores
# Stop during activate and can then fail switch_controller (~90s).
# Use --with-sim-reset only if you manually wait for controllers after Play.
set -euo pipefail

N_DEFAULT="${1:-20}"
RAW_ROOT="${RAW_ROOT:-/data/agent_client_installer/Agent/raw_data}"
OUT_ROOT="${OUT_ROOT:-/data/agent_client_installer/Agent/data_collected}"
LOG_DIR="${LOG_DIR:-/tmp/g1d_auto_collect}"
TUBE_PRIM="${TUBE_PRIM:-/World/room/scene/test_tube}"
START_RACK="${START_RACK:-/World/room/scene/test_tube_rack_01}"
PLACE_RACK="${PLACE_RACK:-/World/room/scene/test_tube_rack}"
DOMAIN_ID="${ROS_DOMAIN_ID:-72}"
RMW="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
WITH_SIM_RESET=0
SETTLE_S="${SETTLE_S:-12}"
CTRL_WAIT_S="${CTRL_WAIT_S:-120}"
HOME_SETTLE_S="${HOME_SETTLE_S:-6}"

ARGS=()
for a in "$@"; do
  case "$a" in
    --with-sim-reset) WITH_SIM_RESET=1 ;;
    --help|-h)
      cat <<'EOF'
Usage: bash scripts/auto_collect_g1d_tube.sh [N] [--with-sim-reset]
  N                 default episode count prompt (default 20); script always asks
  --with-sim-reset  STOP→PLAY between episodes (slow; wait for controllers)

Between episodes: re-enable both arms → FSM HOME → open grippers → teleport tube.
After each take: keep/discard, then continue (y) / quit (q).
EOF
      exit 0
      ;;
    *)
      if [[ "$a" =~ ^[0-9]+$ ]]; then
        N_DEFAULT="$a"
      else
        ARGS+=("$a")
      fi
      ;;
  esac
done

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
# Keep a TTY for prompts even while logging.
exec > >(tee -a "$LOG") 2>&1
TTY="${TTY:-/dev/tty}"

echo "=== semi-interactive collect (sim_reset=$WITH_SIM_RESET) ==="
echo "log=$LOG"
echo "start_rack=$START_RACK  place_rack=$PLACE_RACK  tube=$TUBE_PRIM"

set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
source "$HOME/fa-deploy-ws/install/setup.bash" 2>/dev/null || true
if [[ -f "$HOME/lerobot_ros2/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/lerobot_ros2/.venv/bin/activate"
fi
set -u
export RMW_IMPLEMENTATION="$RMW" ROS_DOMAIN_ID="$DOMAIN_ID" ROS_USE_SIM_TIME=true

prompt() {
  local msg="$1"
  local ans
  printf '%s' "$msg" >"$TTY"
  IFS= read -r ans <"$TTY" || true
  printf '%s\n' "$ans"
}

pub_state() {
  local code="$1"
  ros2 topic pub --once /teleop/controller_state std_msgs/msg/Int32 "{data: ${code}}" >/dev/null
}

wait_stack_ready() {
  # Motion-gen needs latched /robot_description + WBC current_state.
  # Both publishers are TRANSIENT_LOCAL — a VOLATILE late-joiner never sees the
  # cached current_state if WBC is idle in HOLD (that was the false "ocs2_wbc=False").
  CTRL_WAIT_S="$CTRL_WAIT_S" python - <<'PY'
import os, time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from rosidl_runtime_py.utilities import get_message

timeout = float(os.environ.get("CTRL_WAIT_S", "180"))
latch_qos = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)
rclpy.init()
node = Node(
    "collect_wait_stack",
    parameter_overrides=[Parameter("use_sim_time", value=False)],
    automatically_declare_parameters_from_overrides=True,
)
got = {"desc": False, "wbc": False}
node.create_subscription(
    String,
    "/robot_description",
    lambda m: got.__setitem__("desc", bool((m.data or "").strip())),
    latch_qos,
)
try:
    Msg = get_message("arms_ros2_control_msgs/msg/WbcCurrentState")
    node.create_subscription(
        Msg,
        "/ocs2_wbc_controller/current_state",
        lambda _m: got.__setitem__("wbc", True),
        latch_qos,
    )
except Exception as exc:
    print(f"[WARN] cannot subscribe current_state yet: {exc}")

deadline = time.monotonic() + timeout
last_print = 0.0
while time.monotonic() < deadline and not (got["desc"] and got["wbc"]):
    now = time.monotonic()
    if now - last_print > 5.0:
        names = dict(node.get_topic_names_and_types())
        print(
            f"[wait] desc={got['desc']} wbc={got['wbc']} "
            f"topics: robot_description={('/robot_description' in names)} "
            f"current_state={('/ocs2_wbc_controller/current_state' in names)}"
        )
        last_print = now
    rclpy.spin_once(node, timeout_sec=0.2)
node.destroy_node()
rclpy.shutdown()
if not (got["desc"] and got["wbc"]):
    raise SystemExit(
        "FaSim ROS2 stack not ready "
        f"(robot_description={got['desc']}, ocs2_wbc={got['wbc']}).\n"
        "In Isaac: wait until log shows 'Successfully switched controllers' / HOLD.\n"
        "Cold first Play often invalidates PhysX views — Stop once spam ends, wait ~5s, Play again.\n"
        "Do NOT Stop while log says 'ignoring Stop during activate'."
    )
print("[OK] stack ready: /robot_description + /ocs2_wbc_controller/current_state")
PY
}

wait_controllers() {
  wait_stack_ready
}

teleport_tube_to_start_hole() {
  local hole="$1"
  START_RACK="$START_RACK" TUBE_PRIM="$TUBE_PRIM" START_HOLE="$hole" python - <<'PY'
import os, time
from geometry_msgs.msg import Pose
from simulation_interfaces.srv import GetEntityState, SetEntityState
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

rack = os.environ["START_RACK"]
tube = os.environ["TUBE_PRIM"]
hole = os.environ["START_HOLE"]
hole_path = f"{rack}/{hole}"

rclpy.init()
node = Node(
    "collect_teleport_tube",
    parameter_overrides=[Parameter("use_sim_time", value=False)],
    automatically_declare_parameters_from_overrides=True,
)
get_cli = node.create_client(GetEntityState, "/get_entity_state")
set_cli = node.create_client(SetEntityState, "/set_entity_state")
for cli, name in ((get_cli, "get"), (set_cli, "set")):
    if not cli.wait_for_service(timeout_sec=8.0):
        raise SystemExit(f"{name}_entity_state unavailable")

def call(cli, req, timeout=8.0):
    fut = cli.call_async(req)
    start = time.monotonic()
    while not fut.done():
        if time.monotonic() - start > timeout:
            raise SystemExit("service timeout")
        rclpy.spin_once(node, timeout_sec=0.05)
    return fut.result()

from simulation_interfaces.msg import Result

def _ok(res, what: str) -> None:
    # Isaac often fills error_message even on success; trust result code.
    code = int(res.result.result)
    if code != int(Result.RESULT_OK):
        raise SystemExit(
            f"{what} failed: code={code} msg={res.result.error_message!r}"
        )

req = GetEntityState.Request(); req.entity = hole_path
res = call(get_cli, req)
_ok(res, f"get hole {hole_path}")
pose = res.state.pose
# Lift a few mm so physics doesn't explode into the rack mesh, then settle.
pose.position.z += 0.01

sreq = SetEntityState.Request()
sreq.entity = tube
sreq.state.pose = pose
sres = call(set_cli, sreq)
_ok(sres, f"set tube {tube}")
print(f"[OK] teleported {tube} -> {hole_path} xyz=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f})")
# brief settle without Stop/Play
time.sleep(1.0)
node.destroy_node(); rclpy.shutdown()
PY
}

optional_sim_reset() {
  if [[ "$WITH_SIM_RESET" != "1" ]]; then
    return 0
  fi
  echo "[reset] STOP→PLAY (FaSim may take ~90s to re-activate controllers)"
  SETTLE_S="$SETTLE_S" python - <<'PY'
import os, time
from robot_action_composer.isaac_sim import set_simulation_state, wait_entity_pose_readable
settle = float(os.environ.get("SETTLE_S", "12"))
set_simulation_state(0)
time.sleep(0.8)
set_simulation_state(1)
time.sleep(settle)
wait_entity_pose_readable("/World/room/scene/test_tube", max_wait=20.0)
print("[OK] sim reset timeline")
PY
  CTRL_WAIT_S="$CTRL_WAIT_S" wait_controllers
}

reset_arms_home() {
  # pick_tube_full leaves RIGHT_ARM_DISABLE; next take needs both arms from HOME.
  # FSM: HOLD(2) → HOME(1) → settle → HOLD → OCS2(3); then open grippers.
  HOME_SETTLE_S="$HOME_SETTLE_S" python - <<'PY'
import os, time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64, Int32, String

settle = float(os.environ.get("HOME_SETTLE_S", "6"))
FSM_HOME, FSM_HOLD, FSM_OCS2 = 1, 2, 3

rclpy.init()
node = Node(
    "collect_reset_arms_home",
    parameter_overrides=[Parameter("use_sim_time", value=False)],
    automatically_declare_parameters_from_overrides=True,
)
fsm_pub = node.create_publisher(Int32, "/fsm_command", 10)
mode_pub = node.create_publisher(String, "/mode_command", 10)
left_g = node.create_publisher(Float64, "/left_gripper_controller/target_percent", 10)
right_g = node.create_publisher(Float64, "/right_gripper_controller/target_percent", 10)
time.sleep(0.3)  # discovery

def spin_brief(dt=0.05):
    end = time.monotonic() + dt
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)

def pub_fsm(v: int):
    fsm_pub.publish(Int32(data=int(v)))
    spin_brief(0.15)

def pub_mode(cmd: str):
    mode_pub.publish(String(data=cmd))
    spin_brief(0.2)

print("[home] re-enable arms + FSM HOME")
for cmd in ("LEFT_ARM_ENABLE", "RIGHT_ARM_ENABLE", "BASE_LOCK"):
    pub_mode(cmd)
pub_fsm(FSM_HOLD)
pub_fsm(FSM_HOME)
print(f"[home] settling {settle:.1f}s …")
time.sleep(settle)
pub_fsm(FSM_HOLD)
pub_fsm(FSM_OCS2)
for _ in range(3):
    left_g.publish(Float64(data=1.0))
    right_g.publish(Float64(data=1.0))
    spin_brief(0.1)
print("[OK] arms at HOME, grippers open, OCS2")
node.destroy_node()
rclpy.shutdown()
PY
}

soft_qa_raw() {
  local ep_name="$1"
  local raw="$RAW_ROOT/$ep_name"
  python3 - <<PY
import csv, sys
from pathlib import Path
raw = Path("$raw")
ok = True
for side in ("left", "right"):
    p = raw / f"{side}_gripper_command.csv"
    if not p.exists():
        print(f"[FAIL] missing {p.name}"); ok = False; continue
    rows = list(csv.DictReader(open(p)))
    vals = {r["value"] for r in rows}
    ts = [int(r["timestamp_ns"]) for r in rows]
    if not ({"0", "1"} <= vals <= {"0", "1"}):
        print(f"[FAIL] {side} vals={vals}"); ok = False
    elif ts and len(str(ts[0])) >= 18:
        print(f"[FAIL] {side} wall-clock ts0={ts[0]}"); ok = False
    else:
        print(f"[OK] {side} n={len(rows)} vals={sorted(vals)} ts0={ts[0] if ts else None}")
sys.exit(0 if ok else 1)
PY
}

discard_episode() {
  local ep_name="$1"
  rm -rf "$RAW_ROOT/$ep_name" "$OUT_ROOT/$ep_name"
  echo "[discarded] $ep_name"
}

ask_hole() {
  local which="$1"  # start|place
  local default="${2:-3}"
  local ans
  while true; do
    ans="$(prompt "[$which hole 1-6, default $default]: ")"
    ans="${ans:-$default}"
    if [[ "$ans" =~ ^[1-6]$ ]]; then
      printf 'hole_%s\n' "$ans"
      return 0
    fi
    echo "Enter a number 1..6" >"$TTY"
  done
}

ok_count=0
fail_count=0
discard_count=0

ask_n_episodes() {
  local ans
  while true; do
    ans="$(prompt "How many episodes to collect? [default $N_DEFAULT]: ")"
    ans="${ans:-$N_DEFAULT}"
    if [[ "$ans" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "$ans"
      return 0
    fi
    echo "Enter a positive integer" >"$TTY"
  done
}

ask_continue_or_quit() {
  local ans
  while true; do
    ans="$(prompt "Continue collecting? [y=continue / q=quit]: ")"
    case "${ans,,}" in
      y|yes|"") printf 'y\n'; return 0 ;;
      q|quit|n|no) printf 'q\n'; return 0 ;;
      *) echo "Enter y (continue) or q (quit)" >"$TTY" ;;
    esac
  done
}

echo "[0] waiting for FaSim stack (/robot_description + ocs2) — do not collect until this passes"
CTRL_WAIT_S="$CTRL_WAIT_S" wait_stack_ready

N_EPISODES="$(ask_n_episodes)"
echo "=== will collect up to N=$N_EPISODES episodes (quit early with q after any take) ==="

for i in $(seq 1 "$N_EPISODES"); do
  echo
  echo "========== episode $i / $N_EPISODES =========="
  echo "Watch Isaac: you will be asked to KEEP or DISCARD after the motion." >"$TTY"

  start_hole="$(ask_hole start 6)"
  place_hole="$(ask_hole place 3)"
  place_scene="place_${place_hole}"
  echo "plan: start=$start_hole on right rack | place=$place_hole on left rack (scene=$place_scene)"

  optional_sim_reset
  # Re-check before every take (catches mid-session activate failures).
  CTRL_WAIT_S=60 wait_stack_ready || {
    echo "[FAIL] stack lost — Stop, wait 5s, Play, wait for 'controllers activated', then continue"
    fail_count=$((fail_count + 1))
    cont="$(ask_continue_or_quit)"
    [[ "$cont" == "q" ]] && break
    continue
  }

  echo "[1/6] reset arms to HOME (re-enable both arms)"
  if ! reset_arms_home; then
    echo "[WARN] arm HOME reset failed — continuing anyway"
  fi

  echo "[2/6] teleport tube to $START_RACK/$start_hole"
  if ! teleport_tube_to_start_hole "$start_hole"; then
    echo "[FAIL] teleport failed — is Isaac Playing and /set_entity_state up?"
    fail_count=$((fail_count + 1))
    cont="$(ask_continue_or_quit)"
    [[ "$cont" == "q" ]] && break
    continue
  fi

  before=$(ls -1 "$RAW_ROOT" 2>/dev/null | sort || true)

  echo "[3/6] start recording (50)"
  pub_state 50
  sleep 1

  echo "[4/6] motion-generation --scene $place_scene --no-reset"
  if ! motion-generation --robot unitree_g1d --task-key pick_tube_full --scene "$place_scene" --no-reset; then
    echo "[WARN] motion-generation failed"
    pub_state 51 || true
    fail_count=$((fail_count + 1))
    ans="$(prompt "Motion failed. Discard this recording if any [Y/n]: ")"
    # best-effort discard of newest ep
    after=$(ls -1 "$RAW_ROOT" 2>/dev/null | sort || true)
    new_ep=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | grep -E '^[0-9]{8}_' | tail -1 || true)
    if [[ -n "${new_ep:-}" && "${ans:-Y}" != "n" && "${ans:-Y}" != "N" ]]; then
      discard_episode "$new_ep"
      discard_count=$((discard_count + 1))
    fi
    cont="$(ask_continue_or_quit)"
    [[ "$cont" == "q" ]] && break
    continue
  fi

  echo "[5/6] stop recording (51)"
  pub_state 51
  sleep 2

  after=$(ls -1 "$RAW_ROOT" 2>/dev/null | sort || true)
  new_ep=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | grep -E '^[0-9]{8}_' | tail -1 || true)
  if [[ -z "${new_ep:-}" ]]; then
    echo "[FAIL] no new raw episode directory"
    fail_count=$((fail_count + 1))
    cont="$(ask_continue_or_quit)"
    [[ "$cont" == "q" ]] && break
    continue
  fi
  echo "[QA] new episode: $new_ep"
  soft_ok=0
  if soft_qa_raw "$new_ep"; then soft_ok=1; fi

  echo "[6/6] Visual check in Isaac: did pick→handover→place succeed into $place_hole?" >"$TTY"
  ans="$(prompt "Keep this episode? [y/N] (soft_csv_ok=$soft_ok): ")"
  if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
    if [[ "$soft_ok" != "1" ]]; then
      echo "[WARN] kept despite soft CSV QA failure — re-check later"
    fi
    ok_count=$((ok_count + 1))
    echo "[KEEP] $new_ep"
  else
    discard_episode "$new_ep"
    discard_count=$((discard_count + 1))
  fi

  if [[ "$i" -ge "$N_EPISODES" ]]; then
    break
  fi
  cont="$(ask_continue_or_quit)"
  if [[ "$cont" == "q" ]]; then
    echo "[quit] stopping after episode $i (kept=$ok_count)"
    break
  fi
done

echo
echo "=== done: kept=$ok_count discarded=$discard_count motion_fail=$fail_count log=$LOG ==="
echo "Parquet conversion continues in the collector background; do not require it for keep/discard."
