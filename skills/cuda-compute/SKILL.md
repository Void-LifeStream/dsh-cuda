---
name: cuda-compute
description: Run heavy numerical work on the local RTX 4090 through the dsh-cuda tools (cuda_status / cuda_run / cuda_job). Use for matrix math, simulation, Monte Carlo, parameter sweeps, signal or image processing, training small models, or any parallel workload that would take minutes on the CPU. Also use when a CUDA run failed with an out-of-memory error or when you need to know which Python on this machine has torch with CUDA.
whenToUse: The task is numerically heavy, data-parallel, or already slow on the CPU; or you need GPU status before/after a run.
---

# 在这台机器上用 4090 做计算

> 本文件由 2026-10-15 的会话落地，配套插件 `~/.dsh/plugins/dsh-cuda/`（工具）+ 本技能（说明书）。
> 环境实测数据是那天的：驱动 617.42 / torch 2.13.0+cu130 / CUDA UMD 13.4。**环境变了要更新这里。**

## 一、先看有没有得用

| 解释器 | python | torch | CUDA | 用什么 |
|---|---|---|---|---|
| `F:\ComfyUI\ComfyUI-aki-v3.2\python\python.exe` | 3.13.11 | 2.13.0**+cu130** | ✅ | **默认首选**。另有 triton 3.5.1、scipy 1.17、scikit-learn、matplotlib；⚠️ **numba 的 CUDA 后端在本机是废的**（见下） |
| `F:\Stable Diffusion\sd-webui-aki-v4.11.1-cu128\python\python.exe` | 3.11.9 | 2.7.0+cu128 | ✅ | 备选。有 **jax 0.4.38**、pandas；torch 略旧 |
| `~/.dsh/dsh-runtimes/.../python.exe`（DSH 捆绑） | — | ❌ 没有 | ❌ | **只有 numpy/pandas**，别拿它跑 GPU（还要自己装 torch） |

- 显卡 **RTX 4090 / 24 GB / sm_89**；`cuda_run` 默认自动挑第一个可用的解释器，不用你指定。
- ⚠️ **本机没有 nvcc、没有 CUDA Toolkit**（只有要写 CUDA C++ 才需要装，约 3 GB）。
- ⚠️ **写自定义 kernel 只能用 triton —— `numba.cuda` 在本机是废的**（2026-10-15 实测，别重蹈覆辙）：
  `numba.cuda.is_available()` 返回 **False**，任何 kernel 调用都会抛
  `NvvmSupportError: libNVVM cannot be found ... Could not find module 'nvvm.dll'`。
  根因：numba 的 CUDA 后端需要 **libNVVM**（NVIDIA 的 LLVM 编译器库，随 CUDA Toolkit 分发），
  而且 numba 0.61+ 已把 CUDA 支持拆到独立包 `numba-cuda`（本机也没装）。
  **别在它身上浪费时间**，除非用户愿意装数 GB 的 toolkit。
- ✅ **triton 已实测可用**（3.5.1，自带 LLVM/ptxas，零额外安装）：elementwise map / zip / 越界 guard /
  2D / broadcast / 分块大数组 / 规约 dot / 逐行 softmax 八种写法全部跑通，模板脚本见
  `E:\DeepSeek Harness\Infrastructure\cuda-probe\gpu-puzzles\triton_puzzles.py`（可直接抄写法）。
  顺带：开源题库 **`srush/GPU-Puzzles` 的官方 grader 在本机跑不了**（它用 numba.cuda 驱动 kernel），
  上面那个脚本就是把这套题翻译成 triton 版 + 自带 numpy 对拍的产物。
- ⚠️ 没有 cupy。要装得动 ComfyUI 的环境（`<上面第一个路径> -m pip install cupy-cuda13x`）——
  **这会影响用户的 ComfyUI，动手前先问他**。

## 二、三个工具

```
cuda_status  → 显存/占用进程/可用解释器/最近任务      （跑之前先看一眼，跑完 OOM 也看它）
cuda_run     → 提交 Python 代码或脚本文件并在 GPU 上跑
cuda_job     → 跟踪某个任务：logs / wait / kill / info
```

- **代码给 `code=`，长程序用 `script=<路径>.py`**（长脚本先用 write 落盘再跑，别把几百行塞进参数）。
- **预计超过 5 分钟就用 `background=true`**。前台模式超预算**不会杀任务**，而是转后台并返回 job id ——
  这时**千万别重跑**，用 `cuda_job action=wait` 或 `action=logs` 跟着它。
- 每次运行都有自己的目录（脚本副本、`stdout.log`、`stderr.log`、`result.json`），
  路径在工具的 `dir` 字段里；`result.json` 里有**实测峰值显存**和耗时。

## 三、写代码的要点

```python
import os, time, torch
dev = 'cuda'
torch.manual_seed(0)                       # 可复现
x = torch.rand(2_000_000, device=dev)      # 直接建在显存里
y = (x * 3.0).sum()                        # 逐元素 + 规约：吃满带宽
torch.cuda.synchronize()                   # 计时前必须同步
t0 = time.perf_counter()
z = (x * 2.0).sum()
torch.cuda.synchronize()
print(f"{(time.perf_counter()-t0)*1e3:.2f} ms  sum={float(z):.4f}")   # 只打摘要
torch.save({'bad': torch.arange(16, device=dev).cpu()}, os.path.join(os.environ['DSH_CUDA_JOB_DIR'], 'out.pt'))
```

- **显存预算**：`字节 ≈ 元素数 × dtype字节 × 中间张量个数`。fp32 是 4 B、fp64 8 B、bf16 2 B。
  24 GB ≈ 60 亿个 fp32 数（含中间结果要打折）。**超了就在 CPU 上分块喂**（`for chunk in ...: 送显存 → 算 → 取回`）。
- **dtype**：默认 fp32。要快用 `torch.bfloat16`（4090 支持）；要精度才用 fp64（慢一个数量级）。
- **别在循环里 `.item()` / `.cpu()` / `print(张量)`** —— 每次都是同步 + 传输，能把 GPU 饿死。
- **零拷贝**：`torch.from_numpy(arr)` 不复制，`torch.tensor(arr)` 会复制。
- **大数据落地**：`.npy` / `.pt` 写文件，**不要 print 出来**（stdout 超过 12000 字符会被截断，
  还会挤爆上下文）。要回看就用 read 工具读文件。
- **自定义 kernel 用 triton**（`@triton.jit`）—— `numba.cuda` 在本机不可用，别浪费一轮排查（见第一节）。
  八种常用写法的可直接抄模板在
  `E:\DeepSeek Harness\Infrastructure\cuda-probe\gpu-puzzles\triton_puzzles.py`。
  先用小规模对拍 CPU 结果（`np.allclose`）再放大。

## 四、这台机器上的坑（都踩过）

1. **显存是共享的**：ComfyUI（8188）、SD WebUI（7860）、ollama（llama-server）都会占显存。
   跑大任务前用 `cuda_status` 看 free；**不够就先和用户商量停哪个**（他很在意 ollama/生图能不能用）。
2. **`status: running` 不等于还没跑完**：后台任务结束后由 `result.json` 兜底判定，`cuda_job` 会显示
   `done`/`failed`；`lost` 才是真丢了（进程没来得及写事实表就死了）。
3. **失败判据**：`exitCode != 0` 就一定要看 `stderr` —— torch 的报错原文在那里（OOM 会明确写
   `CUDA out of memory`，附已分配/可用显存）。**别猜**。
4. **OOM 的正确反应**：先降 batch/分块/换 bf16，**不要**急着 `torch.cuda.empty_cache()` 骗自己
   （那只是释放缓存，不解决真实占用），更不要重跑一遍一模一样的代码。
5. **中文/编码**：插件已经设了 `PYTHONIOENCODING=utf-8`，中文 print 不会崩；
   但**输出尽量短**，明细写文件。
6. **长任务跨会话存活**：后台任务是 detached 的，**DSH 重启不会杀掉它**。
   所以过几轮对话后你可能看到一个还在跑的老任务 —— 用 `cuda_job action=info` 确认它是什么再决定。
7. **超时预算**：前台默认 300 秒（可用 `timeoutSec` 调，上限 1800 秒）。真跑几小时的任务：
   `background=true` + 中途 `cuda_job action=logs` 看进度（脚本里请定期 `print(..., flush=True)`）。
8. **Windows 的 nvidia-smi 报不出"每个进程占多少显存"**（WDDM 限制）：`cuda_status` 会照实写
   "(WDDM does not report per-process VRAM)" 并只列进程名 —— 别把那张进程表当成显存明细。

## 五、维护这份技能与插件

- **源码都在工作区**：`E:\DeepSeek Harness\Infrastructure\dsh-cuda\`
  （`lib/engine.js` 引擎、`lib/entry.js` 工具与技能注册；本文件在包的 `skills\cuda-compute\SKILL.md`）。
  它 link 装进 desktop profile，改文件即改源；**但 bundle 代码只在 DSH 启动时加载，改完必须重启 DSH**。
- **改完先自测，不用重启**（都用 DSH 捆绑的 node 跑）：
  `test/selftest.mjs` —— 引擎 35 项真机用例；`test/harness.mjs` —— 45 项 mock ctx 用例
  （覆盖工具注册、技能正文加载、真跑一次 GPU 任务、后台与 kill）；
  `test/lossless.mjs` —— 28 项返回值合规用例。
  ⚠️ **mock ctx 测不出 DSH 的"工具返回值必须是 lossless JSON"这条硬规则**（真实失败长这样：
  `Error: tool "cuda_status" returned invalid output: value is not lossless JSON`）——
  返回对象里只要有一个 `undefined`（例如设备正常时的 `gpuError`）就触发，**首轮真机调用正是死在这上面**。
  因此每个工具的返回值都过 `lossless()` 清洗（在 `registerTool` 里统一包），而 `test/lossless.mjs`
  逐字复刻了 `@deepseek-ai/dsh-api-gateway` 的 `visitJsonValue`（拒 `undefined` / `NaN` / `-0` /
  类实例 / 数组空洞 / 循环引用），并带 6 条"故意坏掉"的自检来证明该测试有判别力。
  **改任何工具的返回值之后，务必跑它。**
- **为什么这个插件没有 Config**：link 安装的包解析不到 `@deepseek-ai/*` 内部包（实测会让入口
  `failed to import`），所以它零内部依赖 —— 代价是没有 schemastery 的 Config schema。
  改用环境变量：`DSH_CUDA_PYTHON` / `DSH_CUDA_RUNS_DIR` / `DSH_CUDA_TIMEOUT_MS` /
  `DSH_CUDA_MAX_TIMEOUT_MS` / `DSH_CUDA_MAX_OUTPUT_CHARS`。
- **技能正文改动即时生效**（provider 每次加载重读文件），**只有改工具代码才要重启**。
