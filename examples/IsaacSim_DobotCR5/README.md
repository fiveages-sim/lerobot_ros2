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




https://github.com/user-attachments/assets/d75b44b9-e6a4-4131-bd86-3ea269ea96aa




