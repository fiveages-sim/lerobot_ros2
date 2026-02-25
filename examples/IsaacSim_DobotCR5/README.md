# Using LeRobot ROS2 for Dobot CR5 in Isaac Sim

## Isaac Sim ROS2 环境准备
这里使用的是Isaac Sim 6.0版本

### 配置Isaac Sim以支持使用ROS2 Service来控制仿真的启停
* 参照官网，编译安装
* 参照官网，配置[ros2 workspace](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html#setup-ros-2-workspaces)
* 可以将isaac ros2的内容添加到.bashrc方便后续使用
  ![FiveAges](.images/isaac_ros2.png)

* 接下来，使用[支持仿真控制的方式启动Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_simulation_control.html#enabling-the-extension)
  ![FiveAges](.images/isaac_ros2_sim_control.png)

* 此时，使用Service就可以控制Isaac Sim中仿真的运行，暂停和继续等
  ```bash
  # 启动/继续仿真
  ros2 service call /set_simulation_state simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"  # 1=playing
  ```
  ```bash
  # 重置仿真
  ros2 service call /set_simulation_state simulation_interfaces/srv/SetSimulationState "{state: {state: 0}}"  # 1=playing
  ```

### 配置Isaac 仿真场景以支持使用ROS2 Service来获取并修改仿真中物体的信息

* [配置ROS2 Prim Service ](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_prim_service.html)


### 启动Isaac Sim仿真与ROS2 控制

* Isaac Sim仿真启动
  * 按照[robot_usds](https://github.com/fiveages-sim/robot_usds)中的指引配置好仿真的机器人和环境资产
  * 打开`Grasp_Apple.usd`并运行
* ROS2 Control程序启动
  * 按照[open-deploy-ws](https://github.com/fiveages-sim/open-deploy-ws)中的指引配置好ROS2工作空间
  * 编译好用于dobot cr5的description, 控制器，以及topic based ros2 control
  * 确保可以通过ocs2 arm controller控制仿真中的机器人


https://github.com/user-attachments/assets/78cad128-95e7-475b-828f-d12a9ff4b84e


### 单次抓取放置仿真
```bash
python scripts/grasp_single_demo.py
```
https://github.com/user-attachments/assets/d75b44b9-e6a4-4131-bd86-3ea269ea96aa

### 录制数据集
```bash
python scripts/grasp_record_dataset.py
```

### 将数据集转化为lerobot格式
```bash
python scripts/data_convert_to_lerobot.py
```
默认转换编码为AV1,需要将VLC升级到4.0版本，或者使用cursor可以查看

### 训练模型
```bash
python scripts/train.py
```
或
```bash
python scripts/train.py --policy act --chunk-size 16 --n-action-steps 8   --steps 20000 --batch-size 8  --device cuda
```

### 推理模型
```bash
python scripts/inference.py
```

https://github.com/user-attachments/assets/87a48b3b-db63-41e1-9e7b-be89aa801782


