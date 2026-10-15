# dsh-cuda

Run Python on your local NVIDIA GPU from **DeepSeek Harness**.

Three tools plus one skill, so an agent can do numerical work on the GPU instead of
occupying the CPU: device status, foreground and detached runs, durable logs, and a
measured peak-VRAM figure for every run.

> 中文说明见 [README.zh.md](README.zh.md).

## What you get

| Name | Purpose |
|---|---|
| `cuda_status` | device memory, utilization, GPU processes, every CUDA-capable Python on the machine, recent jobs |
| `cuda_run` | submit Python source (`code=`) or a `.py` file (`script=`, plus `args=`); foreground or `background=true` |
| `cuda_job` | follow one job: `logs` / `wait` / `kill` / `info`, by exact id, unique prefix, or `latest` |
| skill `cuda-compute` | the local terrain, how to write the code, the traps — loaded on demand by the model |

Every run gets its own directory under `~/.dsh/gpu-runs/<job-id>/`:

```
script.py        a copy of what actually ran
_runner.py       the wrapper that records resource facts
stdout.log       full stdout (the tool returns only a tail)
stderr.log       full stderr
job.json         the durable record: pid, cwd, argv, timestamps, status
result.json      written by the wrapper: ok, exit_code, elapsed_s, peak VRAM
```

`peak_alloc_MB` comes from `torch.cuda.max_memory_allocated()` inside the child, so the
number is measured, not estimated. A program can read `DSH_CUDA_JOB_DIR` to write its own
artifacts there — they are listed back to the model automatically.

## Five design decisions worth knowing

1. **Exceeding the foreground budget promotes a run to background instead of killing it.**
   A two-hour simulation should not die because the caller stopped waiting. The result comes
   back with `status: running` and a note; follow it with `cuda_job`. Only an explicit
   cancellation actually terminates.
2. **Background runs are detached** and survive a harness restart — again on purpose,
   because long compute should not be tied to a UI lifecycle. The cost is that "leftover"
   jobs are found by asking, which is what `cuda_status` is for.
3. **A detached run's outcome is settled from `result.json`.** No parent is left to record
   the exit code, so once the pid disappears the wrapper's own fact sheet decides
   `done`/`failed`; only its absence means `lost`.
4. **The skill is registered through a borrowed provider.** The catalog keeps the summary
   and every load re-reads the file, so editing the skill body needs no restart. Skill
   registration is a soft dependency: a deployment without `ctx.skills` still gets the tools.
5. **`lib/engine.js` imports nothing from DSH**, so the engine is driven directly by plain
   Node in the test suite — see below.

## Install

The package lives in a workspace directory. Install it through the harness rather than by
hand-editing the profile (the bundled `cordis-plugin-development` skill explains why):

```
plugin_manager  action: install_bundle  target: <absolute path to this directory>
```

**Bundle code is only loaded at harness startup** — restart DSH after installing or editing.
(The installer's `applied` result means the config was written and the entry imports
cleanly; it does not mean the tools are already visible to a running agent.)

## Configure

Precedence for every setting: explicit plugin config → environment variable →
`~/.dsh/cuda.json` → built-in discovery.

| Environment variable | Meaning | Default |
|---|---|---|
| `DSH_CUDA_PYTHON` | interpreter to use | discovered |
| `DSH_CUDA_RUNS_DIR` | where job directories live | `~/.dsh/gpu-runs` |
| `DSH_CUDA_TIMEOUT_MS` | foreground budget | `300000` |
| `DSH_CUDA_MAX_TIMEOUT_MS` | ceiling for a caller-supplied budget | `1800000` |
| `DSH_CUDA_MAX_OUTPUT_CHARS` | output returned to the model per stream | `12000` |

`~/.dsh/cuda.json` is the convenient place for a machine-specific interpreter (an embedded
Python, a conda env):

```json
{
  "pythonPath": "C:\\path\\to\\python.exe",
  "pythonCandidates": ["C:\\path\\to\\python.exe"],
  "runsDir": "D:\\gpu-runs"
}
```

A malformed file is ignored rather than fatal.

### Interpreter discovery

With nothing configured, the plugin probes every `python`/`python3` on `PATH` and a few
usual install locations, then picks the first candidate that exists. `cuda_status` reports
what each one can actually do (`torch` version, CUDA availability, device name, bf16), and
the probe result is cached for an hour.

An interpreter only counts if it has `torch` **with CUDA**. A system Python without torch
is reported but never chosen. Embedded interpreters shipped with image-generation tools
(ComfyUI, Stable Diffusion WebUI) work well — point `DSH_CUDA_PYTHON` at them.

## Requirements

- DeepSeek Harness, Node 22+
- A CUDA-capable NVIDIA GPU with a driver new enough for the installed torch
- A Python with `torch` + CUDA (the rest — numpy, numba, triton, scipy — is optional)

For **writing custom kernels** the plugin's skill documents what actually works: use
[`triton`](https://github.com/triton-lang/triton) if it is installed. `numba.cuda` needs
`libNVVM` (shipped with the CUDA Toolkit) and fails with
`NvvmSupportError: libNVVM cannot be found` when the Toolkit is absent — a common trap.

## Tests

110 checks across three suites. All of them run with plain Node — no harness restart, no
test framework:

```bash
node test/selftest.mjs    # 36  engine, against the real GPU
node test/harness.mjs     # 46  plugin: mock context drives apply(), then the tools run
node test/lossless.mjs    # 28  every tool value crosses the wire (see below)
```

- `selftest.mjs` covers both run modes, promotion, cancellation, log tails, artifact
  collection and argument validation on real hardware.
- `harness.mjs` builds a fake cordis context with exactly the surface the plugin uses, so
  `apply()` and every tool branch are exercised without the harness.
- `lossless.mjs` deserves its own line: DSH refuses a tool value that is not lossless JSON
  (`visitJsonValue` in `@deepseek-ai/dsh-api-gateway` rejects `undefined`, `NaN`, `-0`,
  class instances, array holes and cycles). A single `undefined` field is enough to make a
  perfectly working tool answer `value is not lossless JSON`, and no mock-based test
  catches it. The suite reimplements that rule verbatim and adds six deliberately broken
  values to prove the check itself has teeth.

## Known limits

- **Windows/WDDM**: `nvidia-smi` reports no per-process VRAM and mixes graphics processes
  into the GPU process list. `cuda_status` says so rather than pretending otherwise.
- **Concurrency**: `cuda_run` is registered as not concurrency-safe, so the harness will not
  overlap two runs; large concurrent jobs would compete for VRAM anyway.
- **No CUDA Toolkit integration**: `nvcc` is never invoked. Everything runs through the
  interpreter, which keeps the plugin zero-install.

## License

MIT — see [LICENSE](LICENSE).
