# 复现本轮结果

从 `/home/rocos/sia/tsr` 运行。依赖相邻 `maniskill_spectrum` 的冻结数据与基线源码；精确路径和 SHA 见 `results/manifest.json`。
本轮环境为 `tsr-dev`，实际库版本已写入 manifest。没有修改生产模型、冻结数据或历史结果。

## 1. 重建完整 Pdiag artifact

已有 36 个文件可直接复用；如需重新拟合，请使用新的输出目录，保留已有结果。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python - <<'PY'
from pathlib import Path
from tests.gtsr._fit_se3_cells import fit_all_pdiag_artifacts
from tests.gtsr._maniskill_data import multiseed_root
out = Path('docs/researchwrite/gtsr-unified-eval-2026-09-30/artifacts/pdiag')
artifacts = fit_all_pdiag_artifacts(multiseed_root(), processes=4, cache_dir=out)
assert len(artifacts) == 36
assert all(a['optimization_success'] for a in artifacts.values())
PY
```

## 2. 运行全部 396 个单元

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python scripts/gtsr_unified_eval.py \
  --out docs/researchwrite/gtsr-unified-eval-2026-09-30/results \
  --artifacts docs/researchwrite/gtsr-unified-eval-2026-09-30/artifacts/pdiag \
  --workers 4
```

缓存使用输入、源码、协议、库版本及全部 artifact 的 SHA，存在输入漂移时拒绝复用。
若更换环境或修改脚本，请建立新的实验目录并复制/调整协议，然后运行；不要覆盖本次证据。
`--aggregate-only` 可在相同输入下重新汇总。

## 3. 重析所有几何尝试并输出表图

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python scripts/gtsr_task_preference.py \
  --out docs/researchwrite/gtsr-unified-eval-2026-09-30/results

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python scripts/gtsr_unified_report.py \
  --out docs/researchwrite/gtsr-unified-eval-2026-09-30
```

报告脚本校验所有源 SHA 和 396 份记录的唯一性/完整性后生成表图及逐 seed 配对差。
图同时保存 PNG、SVG、PDF；所有统计均来自 JSON，未手工改数值。

## 4. 数值检查

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python -m pytest -q \
  tests/gtsr/test_unified_eval.py
```

独立矩阵 log、近 π 与小角度、被动坐标变换、倾斜插入轴上的自由 yaw、零 context 名义锚定，共 5 项检查。
模型运行结果不能由测试通过推断；396 个实际运行单元和全部干预尝试另行记录。
