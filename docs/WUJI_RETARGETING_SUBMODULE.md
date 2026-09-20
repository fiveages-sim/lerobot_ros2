# Wuji Retargeting 子模块（公司仓）

上游：[wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting)  
公司仓：[fiveages-sim/wuji-retargeting](https://github.com/fiveages-sim/wuji-retargeting)

与 HUG 相同：父仓 submodule → 公司仓；运行时优先 `pip install wuji-sdk`。

## 首次发布到公司仓（本机已备好 `company-main`）

```bash
bash scripts/push_wuji_retargeting_company.sh
bash scripts/register_wuji_retargeting_submodule.sh
```

说明：完整上游历史含缺失 LFS 对象（`zhuliang.pkl`），无法直接 push。  
`company-main` 为基于 stub 的 **FF 快照**（去掉 example pkl），无 force。

## 别人 clone

```bash
git submodule update --init submodules/wuji-retargeting
# 不要加 --recursive（mujoco-sim / wuji-description 默认不拉）
pip install 'wuji-sdk>=0.10.0'   # 父仓 .venv
```

## Remotes（子模块内）

- `origin` → `git@github.com:fiveages-sim/wuji-retargeting.git`
- `upstream` → `https://github.com/wuji-technology/wuji-retargeting.git`
