# 本轮验证记录

- 新增9项基准检查，加既有65项canonical嵌入、兼容和表达能力检查：74 passed，8.18 s。
- 新增两份脚本和测试文件的Ruff检查通过；git diff --check通过。
- 1120份逐运行输出SHA逐一复核；全部输入源码、协议SHA不变。
- 全部保存的成功路径通过起终点、连续边及独立密集网格核验。
- h=0的280组方法×seed对照核验通过（包含参考方法自身）。
- 源码与Sprint G scope的git diff为空；未修改生产内核或历史结果。
- 三张图的PNG已目视检查，另有SVG/PDF版本。

测试命令：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/rocos/miniconda3/envs/tsr-dev/bin/python -m pytest -q \
 tests/gtsr/test_geometry_benchmark.py \
 tests/gtsr/test_canonical_embedding.py \
 tests/gtsr/test_canonical_compatibility.py \
 tests/gtsr/test_expressivity.py
```
