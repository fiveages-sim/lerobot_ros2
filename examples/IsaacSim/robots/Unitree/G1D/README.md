# Unitree G1D

双臂人形，末端 Dex1 二指夹爪。当前任务是右手抓起试管。

仿真场景是 FaSim-Isaac 的 `environment/fa-project-usd/Sinopec/2026-10 Test Tube Picking/env/G1D/room.usda`。舞台上的机器人是 `/World/G1D`，试管是 `/World/room/scene/test_tube`。右手夹爪是 `/World/G1D/G1_Arms/right_wrist_yaw_link/right_tcp/Dex1`。

Play 之前把该机器人的 ROS variant 设为 `fiveages`。按下 Play 后，场景里的 `ROS2ControlLaunch` 会自己启动 `robot:=unitree_g1d`、`mode:=full_body` 和夹爪。不要再执行 `ros2-stack launch`，也不要给 `motion-generation` 加 `--ensure-ros2-stack`，否则会再起一套外部运控。

```bash
cd examples/IsaacSim
# 仿真已在 Play，且 ROS=fiveages
motion-generation --robot unitree_g1d --task-key pick_tube --scene lift
```

编排进程需要 `use_sim_time:=true`，时间以仿真发出的 `/clock` 为准。物体位姿走 `/get_entity_state`，prim 是 `/World/room/scene/test_tube`。右手是 `arm: right`，对应 `/World/G1D/.../right_tcp/Dex1`。

这一版任务只抓起试管。抓取点用试管上已有的 `tube_grasp`，位姿仍从刚体 `/World/room/scene/test_tube` 读。末端姿态让夹指指向基座 −X、开合沿 Y、指背朝 +Z，再从机器人这一侧靠近并向上抬起。
