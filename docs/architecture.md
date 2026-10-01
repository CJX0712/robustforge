# RobustForge 架构文档

## 1. 设计原则

- **零重型依赖、可复现、可验证**：核心全部纯 numpy 手写；scikit-learn 仅用于数据集与可选后端；无 torch、无 GPU。
- **接口契约优先**：所有攻击 / 防御 / 评测只依赖 `Classifier` Protocol（`predict` / `predict_proba`，可选 `grad_logits`）。换模型不影响攻击与评测。
- **诚实优先**：旗舰带非劣守护；评测口径统一、可复现；不在文档中粉饰数字。

## 2. 调用单向无环

```
cli → pipeline → {data, defenses, attacks, eval} → core(models)
                       │
                       └─ fusion(RobustFuse-AT) → attacks(pgd) + defenses(base)
```

- `pipeline` 负责数据加载（sklearn digits → 筛选 → 切分）、训练四类防御、公平评测、产出 `PipelineReport`。
- `defenses` 训练模型；`attacks` 生成对抗样本（依赖模型的 `grad_logits` 或数值梯度）。
- `fusion.RobustFuseAT` 复用 `pgd` 与 `NumPyMLP`，不引入新依赖。

## 3. 模块职责

| 模块 | 职责 | 关键不变量 |
|------|------|-----------|
| `core/types` | 数据契约（AttackSpec / BenchmarkEntry / PipelineReport） | 概率列顺序与 `classes_` 一致 |
| `core/errors` | E100–E500 错误分类 | 可程序化处理 |
| `core/config` | `Config` + `RF_*` 环境变量覆盖 | 版本无关、可复现 |
| `core/interfaces` | `Classifier` Protocol | 攻击/评测模型无关 |
| `models/mlp` | 纯 numpy 两层 MLP + **解析输入梯度** | 解析梯度 = 数值梯度（rel<1e-4）；`predict_proba` 行和=1 |
| `models/sklearn_wrap` | 可选 sklearn MLP 后端（缺 `grad_logits` → 攻击退化为数值梯度） | `available()` 探测 |
| `attacks/fgsm` | FGSM（L∞ 单步） | `eps=0` → 原样本 |
| `attacks/pgd` | PGD（L∞/L2 多步最强一阶） | 始终落在 ε 球内 |
| `attacks/cw` | CW-L2（tanh 变量 + c 二分） | 小样本上成功误分类 |
| `defenses/baseline` | 无防御（对照下界） | — |
| `defenses/adv_training` | FGSM-AT / PGD-AT | 复用 MLP 的 `attack_gen` 注入对抗样本 |
| `defenses/smoothing` | 随机平滑（可选，认证半径） | 需大量噪声样本，默认不进 benchmark |
| `fusion/robust_fuse` | **旗舰 RobustFuse-AT** | 非劣守护：最优鲁棒精度 ≥ reference |
| `eval/metrics` | clean / robust / ASR / L∞ 越界 | robust ≤ clean + ε |
| `eval/benchmark` | 单模型跨攻击评测 | 统一口径 |
| `pipeline` | 端到端编排 + 落盘 `benchmark.json` | 固定随机种子保证可复现 |

## 4. 旗舰 RobustFuse-AT 机制

1. **自适应 ε 退火**：`eps_t = eps_max · min(1, (t+1)/warmup)`，前期小 ε 稳定训练，逐步升温，避免强扰动致训崩。
2. **鲁棒性感知早停**：每个 epoch 用确定性 PGD 探针（部署强度 `eps_max`）测验证集鲁棒精度，保存最优鲁棒检查点；连续 `patience` 个 epoch 无提升即停。
3. **非劣守护**：若旗舰最优鲁棒精度低于 reference（标准 PGD-AT 末轮）超过 `guard_margin`，则**直接采用 reference 的参数**，绝不把持平/更差包装成更优。

> 与"混淆梯度"型防御的区别：RobustFuse-AT 的训练目标就是标准 min-max 对抗训练，不遮挡真实梯度；评测使用强 PGD 且固定 `random_start=False` 保证公平可复现。

## 5. 复现与质量门禁

- 单测：`pytest -q -W ignore::UserWarning`（含梯度校验、攻击几何、非劣守护等不变量）。
- 基准：`python -m robustforge.cli run` → `benchmark.json`（固定 `random_state=42`）。
- 静态检查：`ruff check .` / `ruff format --check .`（line-length 96）。

## 6. 性能基线（默认配置）

见 `README.md` 基准表：PGD-AT / RobustFuse-AT 在 ε=0.1 上将鲁棒精度从 0.583（无防御）提升到 0.692；全链路在 CPU 上数秒完成。
