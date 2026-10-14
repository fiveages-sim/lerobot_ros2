# LightNav-0 W2 子模块

公司仓：[fiveages-sim/LightNav-0_W2](https://github.com/fiveages-sim/LightNav-0_W2)  
Submodule 路径：`submodules/LightNav-0_W2`  
URL（`.gitmodules`）：`git@github.com:fiveages-sim/LightNav-0_W2.git`

这是 **W2 LightNav-0 部署胶水**（WebSocket 桥、Quick Start）。不是 [lightorigins/LightNav-0](https://github.com/lightorigins/LightNav-0) 的 fork。上游 `robot_deploy/` 面向 Go2/TRON + Humble，不用于这套 Jazzy W2。

## 别人 clone 后怎么拿到

```bash
git submodule update --init submodules/LightNav-0_W2
```

父仓 gitlink 指向公司仓 `main` 上已推送的 commit。补丁 push 到 `LightNav-0_W2`，再在本仓更新 gitlink。

运行入口：[`submodules/LightNav-0_W2/Quick_Start.md`](../submodules/LightNav-0_W2/Quick_Start.md)。

## 不要和这些混用

| 路径 / 服务 | 原因 |
|-------------|------|
| `submodules/internnav_w2` | DualVLN 用 conda `internnav`、HTTP `:8087`；LightNav 用 conda `lightnav`、WebSocket `:8051` / `:8050`。先停掉另一套再开 |
| 上游 LightNav-0 整仓 | 权重与服务另装，不作为本父仓 submodule |
