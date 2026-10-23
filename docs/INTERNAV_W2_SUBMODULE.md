# InternNav W2 子模块

公司仓：[fiveages-sim/internnav_w2](https://github.com/fiveages-sim/internnav_w2)  
Submodule 路径：`submodules/internnav_w2`  
URL（`.gitmodules`）：`git@github.com:fiveages-sim/internnav_w2.git`

这是 **W2 DualVLN 部署胶水**（桥接脚本、Quick Start、架构说明），不是 [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) 的 fork。模型服务仍单独克隆上游 InternNav，在 GPU 机上跑 HTTP `:8087`。

## 别人 clone 后怎么拿到

```bash
git submodule update --init submodules/internnav_w2
```

父仓 gitlink 指向公司仓 `main` 上已推送的 commit。日常改脚本：在子模块里提交并 push 到 `internnav_w2`，再在本仓更新 gitlink SHA。

运行入口：[`submodules/internnav_w2/Quick_Start.md`](../submodules/internnav_w2/Quick_Start.md)。

## 不要和这些混用

| 路径 / 服务 | 原因 |
|-------------|------|
| `submodules/LightNav-0_W2` | 另一套 GPU 服务（WebSocket `:8051`），不要和 DualVLN 同时占 5090 |
| 上游 InternNav 整仓 | 不作为本父仓 submodule；按子模块 Installation Guide 另装 |
