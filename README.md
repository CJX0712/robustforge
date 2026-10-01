# RobustForge · 对抗鲁棒性 / 对抗机器学习系统

> 纯 numpy 离线兜底核心 · FGSM / PGD / CW-L2 攻击 · 对抗训练(AT) · 旗舰 RobustFuse-AT（非劣守护）
> **作者：晨星** · License：MIT · Python 3.10+

RobustForge 是一套**端到端可运行、可复现、零重型依赖**的对抗鲁棒性实验室。它从零手写了一个 numpy MLP，实现并评测了工业界与学界最主流的对抗攻击（FGSM / PGD / CW-L2）与防御（标准训练 / FGSM-AT / PGD-AT），并以 **RobustFuse-AT** 作为旗舰：自适应 ε 退火 + 鲁棒性感知早停 + **非劣守护**（绝不伪称"更优"）。

> 顶级开源参照：[IBM ART（Adversarial Robustness Toolbox）](https://github.com/Trusted-AI/adversarial-robustness-toolbox) · [cleverhans](https://github.com/cleverhans-lab/cleverhans) · [foolbox](https://github.com/bethgelab/foolbox)。RobustForge 不强制依赖它们，但接口与概念与之对齐；可经 `shap` / `lime` 等后端做交叉验证（可选）。

---

## ✅ 一句话结论

在 digits 子集（3 类 / 400 样本 / ε=0.1）上：**无防御模型鲁棒精度从 0.958 暴跌到 0.583（被攻破近半）；PGD-AT 与旗舰 RobustFuse-AT 把鲁棒精度拉回 0.692**，且旗舰由非劣守护保证**绝不劣于**最强的标准防御 PGD-AT。

---

## 特性

- **攻击全家桶**：FGSM（单步 L∞）、PGD（L∞/L2 多步最强一阶）、CW-L2（Carlini-Wagner，tanh 变量 + c 二分），均带不可变量校验。
- **攻击模型无关**：解析梯度（`grad_logits`）优先，缺失时自动退化为数值梯度 → 任意 `Classifier` 后端可接入。
- **防御族**：`Undefended`（对照下界）、`FGSMAT`、`PGDAT`（Madry et al. 标准强防御）。
- **旗舰 RobustFuse-AT**：自适应 ε 退火稳定训练 + 逐 epoch 鲁棒性感知早停（保存最优鲁棒检查点）+ 非劣守护（低于 reference 超过阈值则诚实退回）。
- **评测严格**：clean / FGSM / PGD 三套口径，攻击固定 `random_start=False` 保证可复现、对照公平（遵循 Athalye et al. 对"混淆梯度"型防御的告诫，绝不使用会掩盖梯度的技巧）。
- **零重型依赖**：核心仅 numpy + scipy + scikit-learn；无 torch、无 GPU 亦可全绿运行。

---

## 安装

```bash
pip install -r requirements.txt
# 开发：pytest + ruff
pip install -e ".[dev]"
```

> 仅需 CPU。`scikit-learn` 仅用于 digits 数据集与可选 sklearn MLP 后端；核心 MLP 与全部攻击/防御均为纯 numpy 手写。

---

## 快速开始

```bash
# 端到端 benchmark（落盘 benchmark.json，默认 3 类 / 400 样本 / 12 epoch）
python -m robustforge.cli run --out benchmark.json

# 调参（也可经环境变量 RF_* 覆盖，见 robustforge/core/config.py）
python -m robustforge.cli run --epochs 20 --n-samples 200 --at-eps 0.2

# 系统信息
python -m robustforge.cli info

# 作为库调用
from robustforge.models.mlp import NumPyMLP
from robustforge.attacks.pgd import pgd
```

---

## 目录结构

```
robustforge/
├── robustforge/
│   ├── core/         # types / errors / config / interfaces（契约层）
│   ├── models/       # mlp.py 纯 numpy MLP（含解析输入梯度）· sklearn_wrap.py 可选后端
│   ├── attacks/      # fgsm / pgd / cw（攻击器，模型无关）
│   ├── defenses/     # baseline（undefended）/ adv_training（FGSM-AT, PGD-AT）/ smoothing（随机平滑，可选）
│   ├── fusion/       # robust_fuse.py 旗舰 RobustFuse-AT
│   ├── eval/         # metrics + benchmark（评测口径）
│   ├── pipeline/     # 端到端编排
│   ├── cli.py        # 命令行入口
│   └── examples/run_demo.py
├── tests/            # pytest 单测（含梯度校验、不可变量、非劣守护）
├── docs/architecture.md
├── benchmark.json    # 默认配置下的真实评测结果
├── requirements.txt / requirements.lock.txt
├── Dockerfile / Makefile / pyproject.toml / .gitignore
└── README.md
```

---

## 方法论（为什么可信）

1. **攻击强度统一**：评测全程用 PGD-20 步（强一阶攻击）做鲁棒精度与 ASR 口径，避免"弱攻击虚高鲁棒"。
2. **可复现对照**：评测 PGD 固定 `random_start=False`，保证不同防御在同一组对抗样本上被公平比较；管线入口统一 `np.random.seed`。
3. **非劣守护（诚实底线）**：旗舰与标准 PGD-AT 在验证集上用确定性 PGD 探针比较鲁棒精度；若旗舰不占优超过 `guard_margin`，则**直接采用 PGD-AT 的参数**，绝不把"持平/更差"包装成"更优"。
4. **不变量自测**：`tests/` 覆盖
   - MLP 解析输入梯度 vs 中心差分数值梯度（rel err < 1e-4）；
   - FGSM(eps=0) 返回原样本、PGD 始终落在 ε 球内；
   - 攻击不会"提升"干净精度（robust ≤ clean）；
   - 旗舰最优鲁棒精度 ≥ reference（非劣守护铁律）；
   - CW 在小样本上确实生成成功对抗样本。

---

## 默认基准结果（digits 子集，3 类 / 400 样本 / ε=0.1）

| 防御 | clean | rob_fgsm | rob_pgd | asr_pgd |
|------|------:|--------:|--------:|--------:|
| undefended   | 0.958 | 0.625 | 0.583 | 0.400 |
| fgsm_at      | 0.717 | 0.625 | 0.617 | 0.140 |
| pgd_at       | 0.950 | 0.708 | 0.692 | 0.272 |
| robustfuse_at| 0.950 | 0.708 | 0.692 | 0.272 |

- **无防御被攻破近半**：robust 0.583 vs clean 0.958，ASR 高达 0.40。
- **FGSM-AT 鲁棒性有限**（干净精度被牺牲，鲁棒仅 0.617）——印证 Madry et al. 结论：单步攻击训练不足以抵御多步 PGD。
- **PGD-AT 是强防御**：鲁棒 0.692，ASR 降到 0.272。
- **RobustFuse-AT 与 PGD-AT 持平**：在非劣守护下旗舰保证"绝不更差"；在更受限数据 / 更长训练下，自适应 ε 与鲁棒早停可进一步避免过拟合、省算力（详见 `docs/architecture.md`）。

> 说明：在样本充足时 PGD-AT 末轮已接近最优，旗舰与其持平属正常；旗舰的增量价值在于**稳定性 + 早停省算力 + 非劣保证**。本仓库追求的是可验证的诚实结论，而非粉饰数字。

---

## 复现命令

```bash
pytest -q -W ignore::UserWarning      # 全绿单测
python -m robustforge.cli run         # 复现上表 benchmark.json
```

---

## License & 作者

MIT · 作者 **晨星**（GitHub: [CJX0712](https://github.com/CJX0712)）。
