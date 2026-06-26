# torch_npu_ops

## triton_npu算子编译流程
依赖
```bash
pip install triton-ascend==3.2.0 pybind11
```
准备，调用triton_npu目录下的setup.py，该步骤会在xllm编译阶段自动执行，可通过手动运行更新triton-ascend的AOT产物
```bash
python3 setup.py
```
生成产物默认在triton_npu/binary目录下

