# dsh-cuda

让 DSH 里的 agent 直接用本机 **NVIDIA CUDA GPU** 跑 Python 计算：数值模拟、矩阵运算、蒙特卡洛、
参数扫描、训练小模型，以及任何本该并行却挤在 CPU 上的活。

> 2026-10-15 落地。含三个工具 + 一份技能（`skills/cuda-compute/SKILL.md`）。
> **不挑卡**：任何 CUDA-capable 的 NVIDIA GPU 都能用（型号 / 显存 / 算力由 `cuda_status` 实时报告，
> 不写死在文档里）。文档里出现的 RTX 4090 数字是作者机器的**实测样本**，不是要求。

## 它给你什么

| 名称 | 作用 |
|---|---|
| `cuda_status` | 显存 / 利用率 / 占着显存的进程 / 本机所有带 CUDA 的 Python / 最近任务 |
| `cuda_run` | 提交 Python 代码或 `.py` 文件在 GPU 上跑（前台或后台） |
| `cuda_job` | 跟踪一个任务：`logs` / `wait` / `kill` / `info` |
| 技能 `cuda-compute` | 本机 CUDA 地形、写法要点、踩过的坑 —— 模型按需加载 |

每次运行都有独立目录（`~/.dsh/gpu-runs/<时间戳-随机>`），里面是脚本副本、`stdout.log`、
`stderr.log`、`job.json` 和 `result.json`；`result.json` 由包装器写入，含**实测峰值显存**、
耗时、退出码 —— 所以工具返回的显存数字是量出来的，不是估的。脚本可以读环境变量
`DSH_CUDA_JOB_DIR` 把产物写进这个目录，工具会自动列出来。

## 设计上值得知道的五件事

1. **前台超时不杀任务，而是转为后台。** 跑了几小时的仿真不该因为调用方停止等待就被销毁。
   返回里会带 `status: running` 和一句 note，接着用 `cuda_job` 跟就行。**显式取消才会真的杀。**
2. **后台任务是 detached 的**，DSH 重启不会带走它们。代价是"遗留任务"要靠 `cuda_status` 才发现 ——
   这是有意的：长算力不该被 UI 生命周期牵着走。
3. **后台任务的状态由 `result.json` 兜底判定。** 没有父进程替它记录退出码，所以 PID 消失后读事实表：
   有就是 `done`/`failed`，没有才是 `lost`。
4. **技能通过 borrowed provider 注册**（与 `dsh-univer-office` 同款）：catalog 只存摘要，
   **每次加载重读文件**，所以改技能正文不需要重启、也不需要缓存失效。
   技能注册是**软依赖** —— 就算部署里没有 `ctx.skills`，三个工具照常工作。
5. **引擎不 import 任何 DSH 包**（`lib/engine.js`），三套自测都能用普通 Node 直接跑，
   **不需要重启 DSH**（共 108 项）：

   ```powershell
   $node = "$env:USERPROFILE\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
   & $node "E:\DeepSeek Harness\Infrastructure\dsh-cuda\test\selftest.mjs"   # 引擎 35 项（真机 GPU）
   & $node "E:\DeepSeek Harness\Infrastructure\dsh-cuda\test\harness.mjs"    # 插件 45 项（mock ctx 驱动 apply）
   & $node "E:\DeepSeek Harness\Infrastructure\dsh-cuda\test\lossless.mjs"   # 返回值合规 28 项
   ```

   ⚠️ `lossless.mjs` 不是凑数的：DSH 要求工具返回值是**无损 JSON**（`visitJsonValue` 拒
   `undefined` / `NaN` / `-0` / 类实例 / 数组空洞 / 循环引用），而首轮真机调用正是死在一个
   `gpuError: undefined` 上 —— mock 测试测不出这条，所以它被单独复刻成测试。**改任何工具的
   返回值之后都要跑它。**

## 安装

包放在工作区（**不要**建在 `$DSH_HOME` 下，也**不要**手改 profile 的 `package.json` / `cordis.patch.yml` ——
那是 DSH 自带技能 `cordis-plugin-development` 明确劝阻的做法）：

```
plugin_manager  action: install_bundle  target: E:\DeepSeek Harness\Infrastructure\dsh-cuda
```

`install_bundle` 会替你写 profile 的依赖与 bundle 清单并跑 pnpm。**host 插件需要重启 DSH 才加载。**

## 配置（可选）

在 profile 的 `cordis.patch.yml` 里覆盖（或经 GUI 的插件详情页）：

```yaml
- id: dsh-cuda
  config:
    pythonPath: 'F:\ComfyUI\ComfyUI-aki-v3.2\python\python.exe'  # 留空 = 自动挑第一个可用候选
    pythonCandidates: []          # 追加候选解释器
    runsDir: ''                   # 默认 ~/.dsh/gpu-runs
    defaultTimeoutMs: 300000      # 前台预算（超了转后台）
    maxTimeoutMs: 1800000         # 前台预算上限
    maxOutputChars: 12000         # 每次回给模型的输出上限
```

## 作者机器的实测样本（2026-10-15）

> 下面是**样本**，不是要求 —— 你的机器请以 `cuda_status` 的输出为准。

- RTX 4090 / 24564 MiB / sm_89，驱动 617.42；fp32 matmul 实测约 58 TFLOPS
- 可用解释器：ComfyUI 的 `python 3.13.11 + torch 2.13.0+cu130`（另有 numba / triton / scipy / sklearn），
  SD WebUI 的 `python 3.11.9 + torch 2.7.0+cu128`（另有 jax / pandas）
- **没有 nvcc、没有 CUDA Toolkit、没有 cupy** —— 写 CUDA C++ 才需要 toolkit；
  ⚠️ **`numba.cuda` 在本机用不了**（`libNVVM cannot be found`，libNVVM 随 Toolkit 分发，且 numba 0.61+
  已把 CUDA 拆到独立的 `numba-cuda` 包）—— **写自定义 kernel 用 triton**（3.5.1，自带 LLVM/ptxas，
  实测 map/zip/guard/2D/broadcast/blocks/dot/softmax 八种写法全通，模板见
  `cuda-probe\gpu-puzzles\triton_puzzles.py`；那套 `srush/GPU-Puzzles` 的官方 grader 因为依赖 numba
  在本机跑不了，这份就是把题目翻译成 triton 版的产物）
- DSH 捆绑的 Python 只有 numpy/pandas，**没有 torch**

## 许可

MIT。
