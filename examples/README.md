# Examples

基于Lerobot和ROS2的实例代码。

## 项目结构

本项目包含十个实例：

1. **examples/dataset_info.py** - 读取数据集信息
    ```bash
    python examples/dataset_info.py /home/xxxx/lerobot_ros2/dataset/grasp_dataset_1767607004
    ```
2. **examples/demo_control.py** - 可以对已训练好的基于lerobot的模型进行仿真或者实物推理
    ```bash
    CUDA_VISIBLE_DEVICES=0 python examples/demo_control.py --dataset dataset/grasp_dataset_50_no_depth --train-config outputs/act_run2/checkpoints/last/pretrained_model/train_config.json --checkpoint outputs/act_run2/checkpoints/last --device cuda
    ```
3. **examples/demo_grasp.py** - 自动抓取实例代码
    ```bash
    python examples/demo_grasp.py
    ```
4. **examples/demo_move.py** - 机械臂移动实例代码
    ```bash
    python examples/demo_move.py
    ```
5. **examples/demo_record.py** - 简易自动抓取并采集数据实例代码
    ```bash
    python examples/demo_record.py
    ```
6. **examples/grasp_record.py** - 完整的自动抓取并采集数据实例代码（可以选择是否采集深度图像和关键点信息）
    ```bash
    python examples/grasp_record.py
    ```
7. **examples/infer_act.py** - 用于测试训练好的模型
    ```bash
    CUDA_VISIBLE_DEVICES=1 python examples/infer_act.py /home/xxx/outputs/act_overfit1/checkpoints/last --dataset /home/xxx/lerobot_ros2/dataset/grasp_dataset_1 --num-episodes 1
    ```
8. **examples/train_demo.py** - 用于训练基于Lerobot的模型
    ```bash
    CUDA_VISIBLE_DEVICES=1 python examples/train_demo.py  /data/sylcito/grasp_dataset_50  --policy act --chunk-size 16 --n-action-steps 8   --steps 20000 --batch-size 8   --output-dir outputs/act_run1 --device cuda
    ```
9. **examples/visualize_dataset_v2.py** - 用于可视化读取基于Lerobot的数据集
    ```bash
    python examples/visualize_dataset_v2.py /path/to/dataset --episode-index 0 --batch-size 8
    ```
10. **examples/visualize_pcd.py** - 用于可视化采集的3D点云文件
    ```bash
    python examples/visualize_pcd.py --root /home/king/lerobot_ros2/dataset/grasp_dataset_1 --episode 0 --frame 0 --stride 1
    ```


