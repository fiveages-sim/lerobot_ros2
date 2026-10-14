# InternNav ARX Lift2S 子模块

公司仓：[fiveages-sim/internnav_arx_lift2s](https://github.com/fiveages-sim/internnav_arx_lift2s)  
Submodule 路径：`submodules/internnav_arx_lift2s`  
URL（`.gitmodules`）：`git@github.com:fiveages-sim/internnav_arx_lift2s.git`

这是 **Lift2S DualVLN 部署胶水**（RealSense 采帧、HTTP smoke、可选 `/cmd_vel` 桥）。不是 InternNav 整仓 fork。底盘 / 控制在 [fiveages-sim/open-deploy-ws](https://github.com/fiveages-sim/open-deploy-ws) 分支 `arx-lift2s`。

## 别人 clone 后怎么拿到

```bash
git submodule update --init submodules/internnav_arx_lift2s
```

父仓 gitlink 指向公司仓 `main` 上已推送的 commit。补丁 push 到 `internnav_arx_lift2s`，再在本仓更新 gitlink。

运行入口：[`submodules/internnav_arx_lift2s/docs/Quick_Start.md`](../submodules/internnav_arx_lift2s/docs/Quick_Start.md)（完整干跑见同目录 `Basic_Dry_Run_and_Motion_Guide.md`）。

## 不要和这些混用

| 路径 / 服务 | 原因 |
|-------------|------|
| `submodules/internnav_w2` | W2 头相机与桥，话题和机位不同 |
| 上游 InternNav 整仓 | 推理服务另装，不嵌进本父仓 |
