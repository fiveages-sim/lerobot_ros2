#!/usr/bin/env python3
"""Recording policy config for Dobot CR5 IsaacSim dataset scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordConfig:
    fps: int = 30
    camera_info_timeout: float = 10.0
    task_name: str = "pick_place_flow"
    enable_keypoint_pcd: bool = True
    include_depth_feature: bool = False
    # LeRobot dataset write/encode tuning (does not change frame alignment semantics)
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    video_encoding_batch_size: int = 5


RECORD_CFG = RecordConfig()
