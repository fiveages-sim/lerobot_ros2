"""Ubtech Cruzr S2 handover task config."""

TASK_CONFIG = {
    # 任务唯一键：用于代码里索引/选择任务（建议稳定且唯一，通常与文件名一致）
    "task_key": "handover",
    # 展示用名称：用于 UI/日志，不影响逻辑
    "label": "Handover",
    # 任务类型：框架根据它加载对应的任务逻辑/状态机（必须是代码支持的 kind）
    "kind": "handover",
    # 机器人实例/配置 ID：需要与项目中注册的 robot profile 名称匹配
    "robot_id": "ubtech_cruzr_s2_bimanual_handover",
    # 默认场景预设：从 scene_presets 里选择一个 key
    "default_scene": "grab_medicine",
    "base_task_overrides": {
        # 初始抓取用哪只手：left/right。通常选离物体更近、路径更少碰撞的一侧。
        "initial_grasp_arm": "right",
        # 抓取时末端目标姿态（四元数 x,y,z,w）。抓取总“拧腕/歪”优先调这里。
        "grasp_orientation": (0.5, 0.5, -0.5, 0.5),
        # "grasp_orientation": (1.0, 0.0, 0.0, 0.0),
        # 物体初始位置随机扰动（米，x,y,z）。想先跑通流程设为 0；做泛化再逐步加到 1~3cm。
        "object_xyz_random_offset": (0.0, 0.0, 0.0),
        # 接近安全距离（米）：抓取前先移到沿抓取方向偏移此距离的预备位，避免直接撞向物体。
        # 例如目标点 z=1.0、方向向上，则预备位 z=1.0+0.1=1.1，夹爪张开后再下移。
        "approach_clearance": 0.1,
        # 最终夹持距离（米）：沿抓取方向在目标点基础上的偏移，负值表示超过目标点继续深入。
        # 例如 -0.01 表示末端插入目标中心以下 1cm 再闭合夹爪，防止从物体边缘滑过。
        "grasp_clearance":0.0,
        # 夹持位附加微调（米，x,y,z）：叠加在 grasp_clearance 位置上的额外偏移。
        # 用于补偿夹爪 TCP 与实际夹持点之间的机械偏差。
        # 例如 (0,0,-0.08) 表示 TCP 到达目标后再沿 z 下移 8cm，对应指尖在 TCP 下方 8cm 的夹爪结构。
        "grasp_offset": (0.06, 0.0, 0.03),
        "receiver_handover_offset": (-0.025, 0.0, -0.06),
        "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        # 交接空间位置（米，世界坐标 x,y,z）。应保证两只手都可达且远离桌面/身体。
        "handover_position": (0.5, 0.005, 1.3),
        # 交接时递交方末端姿态（四元数 x,y,z,w）。决定“递”物体的朝向/夹爪朝向。
        "source_handover_orientation": (0.6533, 0.2706, 0.2706, 0.6533),
        # 交接时接收方末端姿态（四元数 x,y,z,w）。接收经常夹不到/夹歪通常优先调它+receiver_handover_offset。
        "receiver_handover_orientation": (0.6533, -0.2706, -0.2706, 0.6533),
        # 接收成功后放置位置（米，世界坐标 x,y,z）。z 通常应为“桌面高度 + 安全裕量”。
        "receiver_place_position": (0.5, 0.28, 1.0),
        # 放置时末端姿态（四元数 x,y,z,w）。用于保证放置时夹爪不扫到桌面/物体。
        "receiver_place_orientation": (
            0.6533, -0.2706, -0.2706, 0.6533,
        ),
    },
    "task_queue": [
        {"skill": "handover.pregrasp"},
        {"skill": "handover.pick"},
        {"skill": "handover.exchange_place"},
        {"skill": "handover.movej_return_initial"},
    ],
    "scene_presets": {
        # 场景预设：用于按场景切换被抓取物体（以及你后续可以扩展更多参数覆盖）
        "grab_medicine": {
            "source_object_entity_path": "/World/medicine_handover/FinasterideTablets/tablets/tablets",
        },
        "grab_bottle": {
            "source_object_entity_path": "/World/ConvenienceStore01/SM_Bottle_04_85",
        },
    },
    "record": {
        "base_record_overrides": {
            # 录制数据的任务名标签（建议与 task_key 一致，便于检索/训练）
            "task_name": "handover",
            # 录制帧率：30 常用；降低可减小数据量；提高更细但写盘/编码压力更大
            "fps": 30,
            # 等待相机 CameraInfo/内参的超时（秒）。相机节点启动慢可适当调大。
            "camera_info_timeout": 10.0,
            # 每个 episode 结束后是否切到“保持/定住”模式，避免漂移影响下一回合
            "switch_to_hold_after_episode": True,
            # 是否异步保存 episode（录制与写盘/编码并行）。吞吐更高，但需要配合队列大小。
            "async_episode_save": True,
            # 异步保存队列长度：磁盘/编码慢可增大；太大更占内存且异常时会丢更多未落盘数据
            "episode_save_queue_size": 2,
            # 是否生成关键点点云（通常较吃算力/存储）。不用就关。
            "enable_keypoint_pcd": False,
            # 是否包含深度相关特征。需要 3D/深度训练才开，否则关省空间。
            "include_depth_feature": False,
            # 图像写盘进程数：0 通常表示不额外启多进程（具体语义取决于实现）
            "image_writer_processes": 0,
            # 图像写盘线程数：NVMe 可适当加大；机械盘/网络盘建议小一些避免抖动
            "image_writer_threads": 8,
            # 视频编码批量大小：更大可能吞吐更好，但延迟/内存更高
            "video_encoding_batch_size": 5,
        },
        "profiles": [
            {
                # 配置档位 key：运行时选择 profile 用
                "key": "default",
                # 展示名
                "label": "Default",
                # 覆盖 base_record_overrides 的局部参数
                "overrides": {},
            },
            {
                "key": "fast",
                "label": "Fast Encode Queue",
                "overrides": {
                    # 更高吞吐：增加写盘线程、增大编码 batch（需要足够 CPU/磁盘带宽）
                    "image_writer_threads": 12,
                    "video_encoding_batch_size": 10,
                },
            },
        ],
    }
}
