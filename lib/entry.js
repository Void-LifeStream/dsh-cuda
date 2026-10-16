/**
 * dsh-cuda — run Python work on the local CUDA GPU.
 *
 * Three tools, deliberately small: every schema is priced into every request.
 * - `cuda_status`  what the device and the interpreters look like right now
 * - `cuda_run`     submit Python source (or a script file) and get the result
 * - `cuda_job`     follow, wait for, or kill one job
 *
 * Plus one skill (`skills/cuda-compute/SKILL.md`) registered through a borrowed
 * provider: the catalog keeps the summary and every load re-reads the file, so
 * editing the body needs no restart.
 *
 * ZERO DSH imports, on purpose. Measured 2026-10-15: a `link:`-installed plugin
 * whose real path lives outside the profile cannot resolve `@deepseek-ai/*`
 * packages (the harness resolves those itself for packages inside its own
 * tree); `import { defineTool } from '@deepseek-ai/dsh-tools'` made the whole
 * entry fail with `failed to import` while an entry with no such import
 * activated fine. So this file talks to `ctx.tools` / `ctx.skills` directly and
 * declares parameters as the raw JSON Schema that `defineTool` would have
 * produced anyway. The upside: no dependency on harness internals, and editing
 * this package never needs a reinstall.
 *
 * The engine lives in ./engine.js and imports nothing from DSH either, so
 * `test/selftest.mjs` drives it with plain Node — no restart required.
 */
import { appendFile, mkdir, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GpuEngine } from './engine.js';

export const name = 'dsh-cuda';

/** The tool registry is a hard dependency; the skill registry is picked up when present. */
export const inject = ['tools'];

/**
 * Registration diagnostics. A plugin can report `applied` while every
 * `ctx.tools.register` call inside `apply` still failed, and the failure is
 * invisible from the outside — so record what actually happened on disk.
 */
function pluginLog(line) {
  const dir = process.env.DSH_HOME || join(homedir(), '.dsh');
  const file = join(dir, 'gpu-runs', '_plugin.log');
  return mkdir(join(dir, 'gpu-runs'), { recursive: true })
    .then(() => appendFile(file, `${new Date().toISOString()} ${line}\n`, 'utf8'))
    .catch(() => {});
}

function describeError(error) {
  if (!error) return 'unknown error';
  const name = error.name ?? 'Error';
  const message = error.message ?? String(error);
  const stack = typeof error.stack === 'string' ? error.stack.split('\n').slice(1, 3).join(' | ') : '';
  return `${name}: ${message}${stack ? ` @ ${stack}` : ''}`;
}

/* ------------------------------------------------------------------ *
 * The bundled skill
 * ------------------------------------------------------------------ */

const SKILL_NAME = 'cuda-compute';
const SKILL_URL = new URL(`../skills/${SKILL_NAME}/SKILL.md`, import.meta.url);
const SKILL_DIR = fileURLToPath(new URL(`../skills/${SKILL_NAME}/`, import.meta.url));
const SKILL_PROVIDER = 'dsh-cuda';
const SKILL_INVOCATION = { modelInvocable: true, userInvocable: true };
/** Rank a bundled root uses, so preset skills still win a name collision. */
const SKILL_RANK = 600;

const SKILL_SUMMARY = {
  name: SKILL_NAME,
  description:
    'Run heavy numerical work on the local NVIDIA CUDA GPU through the dsh-cuda tools (cuda_status / cuda_run / cuda_job). ' +
    'Use for matrix math, simulation, Monte Carlo, parameter sweeps, signal or image processing, training small models, ' +
    'or any parallel workload that would take minutes on the CPU; also when a CUDA run failed with an out-of-memory ' +
    'error or when you need to know which Python on this machine has torch with CUDA.',
  whenToUse: 'The task is numerically heavy, data-parallel, or already slow on the CPU; or you need GPU status before or after a run.',
};

const skillProvider = {
  name: SKILL_PROVIDER,
  list: () => Promise.resolve([{
    ...SKILL_SUMMARY,
    invocation: SKILL_INVOCATION,
    provider: SKILL_PROVIDER,
    source: 'bundled',
    resourceBase: { kind: 'directory', path: SKILL_DIR },
    rank: SKILL_RANK,
    locator: SKILL_URL,
  }]),
  async get(candidate) {
    const locator = candidate?.locator instanceof URL ? candidate.locator : SKILL_URL;
    return {
      name: SKILL_SUMMARY.name,
      description: SKILL_SUMMARY.description,
      whenToUse: SKILL_SUMMARY.whenToUse,
      invocation: SKILL_INVOCATION,
      provider: SKILL_PROVIDER,
      source: 'bundled',
      resourceBase: { kind: 'directory', path: SKILL_DIR },
      content: stripFrontmatter(await readFile(locator, 'utf8')),
    };
  },
};

function stripFrontmatter(value) {
  if (!value.startsWith('---\n')) return value;
  const end = value.indexOf('\n---\n', 4);
  return end === -1 ? value : value.slice(end + 5);
}

/* ------------------------------------------------------------------ *
 * Formatting helpers
 * ------------------------------------------------------------------ */

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function formatMs(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '?';
  if (ms < 1_000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function shortPath(path, home) {
  return typeof path === 'string' && home && path.startsWith(home) ? `~${path.slice(home.length)}` : String(path ?? '');
}

/** The calling session's workspace, so relative data paths behave as expected. */
function agentCwd(exec) {
  const cwd = exec?.agent?.session?.header?.cwd;
  return typeof cwd === 'string' && cwd !== '' ? cwd : process.cwd();
}

function numberArg(value, fallback, min, max) {
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

/* ------------------------------------------------------------------ *
 * Plugin
 * ------------------------------------------------------------------ */

/**
 * DSH refuses a tool value that is not lossless JSON. The exact rule lives in
 * `@deepseek-ai/dsh-api-gateway` (`visitJsonValue`): only null, string, boolean
 * and finite non-negative-zero numbers; arrays and plain objects only, no
 * `undefined`, no class instances, no cycles. A tool that returns a raw engine
 * object trips it — `gpuError: undefined` alone is enough. Every definition
 * registered below therefore returns through this scrubber.
 * @param {unknown} value - any JSON-ish value.
 * @returns {unknown} the same shape, representable on the wire.
 */
function lossless(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) {
    // Array.from keeps holes as undefined, which the element scrubber turns into null.
    return Array.from(value, (entry) => lossless(entry === undefined ? null : entry));
  }
  if (typeof value !== 'object') return null;
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    // A class instance (Date, URL, Map…) has no lossless form; keep it readable.
    return typeof value.toString === 'function' ? String(value) : null;
  }
  const out = {};
  for (const [key, entry] of Object.entries(value)) {
    if (entry === undefined) continue;
    out[key] = lossless(entry);
  }
  return out;
}

/**
 * Register one tool through an owned effect, and record the outcome.
 * @param {object} ctx - the plugin context.
 * @param {object} definition - a registry-ready tool definition.
 */
function registerTool(ctx, definition) {
  const run = definition.execute;
  const scrubbed = {
    ...definition,
    async execute(args, exec) {
      return lossless(await run(args, exec));
    },
  };
  ctx.effect(() => {
    try {
      const dispose = ctx.tools.register(scrubbed);
      pluginLog(`${definition.name}: registered`);
      return dispose;
    } catch (error) {
      pluginLog(`${definition.name}: REGISTER FAILED ${describeError(error)}`);
      throw error;
    }
  }, `dsh-cuda: ${definition.name}`);
}

/**
 * Register the CUDA tools.
 * @param {object} ctx - the plugin context (typed by cordis; not imported here).
 * @param {object} [config] - optional configuration; environment variables fill the same fields.
 */
export function apply(ctx, config) {
  const engine = new GpuEngine(config ?? {});

  try {
    const keys = Object.keys(ctx).filter((key) => !key.startsWith('_')).join(',');
    pluginLog(
      `apply: tools.register=${typeof ctx.tools?.register} skills.registerProvider=${typeof ctx.skills?.registerProvider} keys=[${keys}]`,
    );
  } catch (error) {
    pluginLog(`apply: introspection failed ${describeError(error)}`);
  }

  // The skill registry is an optional seam: a deployment without `ctx.skills`
  // still gets the tools, just without the bundled instructions.
  ctx.inject(['skills'], (skillsCtx) => {
    pluginLog('skills: inject fired');
    try {
      skillsCtx.effect(() => {
        try {
          const dispose = skillsCtx.skills.registerProvider(() => skillProvider);
          pluginLog('skills: provider registered');
          return dispose;
        } catch (error) {
          pluginLog(`skills: PROVIDER FAILED ${describeError(error)}`);
          throw error;
        }
      }, 'dsh-cuda: cuda-compute skill');
    } catch (error) {
      pluginLog(`skills: inject callback failed ${describeError(error)}`);
    }
  });

  /* --------------------------- cuda_status -------------------------- */
  registerTool(ctx, {
    name: 'cuda_status',
    description:
      'Inspect the local CUDA GPU before or after running work: device memory, utilization, the processes holding VRAM, ' +
      'every CUDA-capable Python interpreter on this machine, and recent GPU jobs. Call it before a heavy run, when a run ' +
      'failed with CUDA out of memory, or when you need to know which Python has torch+CUDA.',
    parameters: {
      type: 'object',
      properties: {
        jobs: { type: 'number', description: 'How many recent jobs to list (default 5).' },
        refresh: { type: 'boolean', description: 'Re-probe the interpreters instead of using the cached result.' },
      },
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: renderStatus(value, engine.runsDir) }],
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const [device, interpreters, jobs] = await Promise.all([
        engine.devices({ signal: exec?.signal }).catch((error) => ({
          error: String(error?.message ?? error),
          gpus: [],
          processes: [],
        })),
        engine.interpreters({ refresh: args?.refresh === true, signal: exec?.signal }),
        engine.listJobs({ limit: numberArg(args?.jobs, 5, 1, 20) }),
      ]);
      return {
        runsDir: engine.runsDir,
        gpus: device.gpus ?? [],
        gpuError: device.error,
        processes: device.processes ?? [],
        interpreters,
        jobs,
      };
    },
  });

  /* ---------------------------- cuda_run ---------------------------- */
  registerTool(ctx, {
    name: 'cuda_run',
    description:
      'Run Python on the local NVIDIA CUDA GPU — any card your torch build supports; cuda_status reports its memory, ' +
      'compute capability and which interpreters have torch with CUDA (triton is there for custom kernels). Use it for ' +
      'numerical simulation, matrix and array math, parameter sweeps, Monte Carlo, training, or any parallel workload that ' +
      'should not run on the CPU. Every run gets its own directory with the script, logs and a factual summary (peak VRAM, ' +
      'elapsed time, exit code); the program can read DSH_CUDA_JOB_DIR to write artifacts there. Set background=true for work ' +
      'expected to take more than a few minutes; a foreground run that exceeds its budget keeps running in the background and ' +
      'returns its job id.',
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'Python source to execute. Use this for short, self-contained programs.' },
        script: { type: 'string', description: 'Path to an existing .py file to run instead of `code`. Use this for long programs written with the file tools.' },
        args: {
          type: 'array',
          items: { type: 'string' },
          description: 'Command-line arguments passed to the program as sys.argv[1:]. Use this to sweep parameters, e.g. args: ["--alpha", "0.5"].',
        },
        background: { type: 'boolean', description: 'Start detached and return immediately, then follow it with cuda_job. Default false.' },
        timeoutSec: { type: 'number', description: 'Foreground budget in seconds (default 300). Ignored in background mode.' },
        cwd: { type: 'string', description: 'Working directory for the program; defaults to the session workspace. Relative data paths resolve from here.' },
        name: { type: 'string', description: 'Short label recorded with the job, for later recognition.' },
      },
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: renderJob(value, engine.runsDir) }],
    },
    timeoutMs: engine.maxTimeoutMs + 120_000,
    async execute(args, exec) {
      const cwd = typeof args?.cwd === 'string' && args.cwd.trim() !== '' ? args.cwd.trim() : agentCwd(exec);
      const label = typeof args?.name === 'string' ? args.name : undefined;
      const argList = Array.isArray(args?.args) ? args.args.map((entry) => String(entry)) : undefined;
      if (args?.background === true) {
        const started = await engine.runBackground({ code: args.code, script: args.script, cwd, name: label, args: argList });
        return {
          job: started.id,
          status: 'running',
          mode: 'background',
          pid: started.pid,
          cwd,
          dir: started.dir,
          note:
            'Started detached. It keeps running if the harness restarts; follow it with cuda_job ' +
            '(action "wait" to block, "logs" to peek, "kill" to stop).',
        };
      }
      return engine.runForeground({
        code: args?.code,
        script: args?.script,
        args: argList,
        cwd,
        name: label,
        timeoutMs: args?.timeoutSec === undefined ? undefined : Math.round(Number(args.timeoutSec) * 1000),
        signal: exec?.signal,
      });
    },
  });

  /* ---------------------------- cuda_job ---------------------------- */
  registerTool(ctx, {
    name: 'cuda_job',
    description:
      'Follow up on one GPU job started by cuda_run: read its output, wait for it to finish, or terminate it. ' +
      'Pass the job id, a unique prefix, or `latest`.',
    parameters: {
      type: 'object',
      properties: {
        job: { type: 'string', description: 'Job id, unique id prefix, or `latest`.' },
        action: {
          type: 'string',
          enum: ['logs', 'wait', 'kill', 'info'],
          description: 'logs (default) reads output; wait blocks until it ends; kill terminates the process tree; info reports status only.',
        },
        stream: { type: 'string', enum: ['both', 'stdout', 'stderr'], description: 'Which stream logs to return (default both).' },
        tailChars: { type: 'number', description: 'Characters of each stream to return (default 8000).' },
        waitMs: { type: 'number', description: 'For action=wait: how long to block in ms (default 600000).' },
      },
      required: ['job'],
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: renderJob(value, engine.runsDir) }],
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const job = typeof args?.job === 'string' && args.job.trim() !== '' ? args.job.trim() : 'latest';
      const action = args?.action ?? 'logs';
      if (action === 'kill') return { action, ...(await engine.kill(job)) };
      if (action === 'wait') {
        const waited = await engine.wait(job, {
          timeoutMs: numberArg(args?.waitMs, 600_000, 1_000, 3_600_000),
          signal: exec?.signal,
        });
        const described = await engine.describeJob(waited.dir, waited.record, { logLimit: engine.maxOutputChars });
        return { action, stillRunning: waited.stillRunning === true, ...described };
      }
      if (action === 'info') {
        const { dir, record, alive } = await engine.status(job);
        return {
          action,
          job: record.id ?? job,
          name: record.name ?? null,
          status: record.status,
          mode: record.mode ?? 'foreground',
          pid: record.pid ?? null,
          alive,
          cwd: record.cwd ?? null,
          dir,
          startedAt: record.startedAt ?? null,
          finishedAt: record.finishedAt ?? null,
          exitCode: record.exitCode ?? null,
          elapsedMs: record.elapsedMs ?? null,
        };
      }
      const logs = await engine.logs(job, {
        stream: args?.stream ?? 'both',
        tailChars: numberArg(args?.tailChars, 8_000, 500, 60_000),
      });
      return {
        action: 'logs',
        job: logs.job,
        name: logs.name ?? null,
        status: logs.status,
        mode: logs.mode ?? 'foreground',
        exitCode: logs.exitCode ?? null,
        cwd: logs.cwd ?? null,
        dir: logs.dir,
        stdout: logs.stdout,
        stderr: logs.stderr,
      };
    },
  });
}

/* ------------------------------------------------------------------ *
 * Rendering
 * ------------------------------------------------------------------ */

function renderStatus(value, runsDir) {
  const lines = [];
  const gpus = value.gpus ?? [];
  if (value.gpuError) lines.push(`nvidia-smi failed: ${value.gpuError}`);
  for (const [index, gpu] of gpus.entries()) {
    lines.push(
      `GPU ${index}  ${gpu.name}  driver ${gpu.driver}  ${gpu.computeCapability}`,
      `  memory ${gpu.memoryUsedMiB} / ${gpu.memoryTotalMiB} MiB used (free ${gpu.memoryFreeMiB})` +
        `  util ${gpu.utilizationPct}%  temp ${gpu.temperatureC}C  power ${gpu.powerW}W`,
    );
  }
  if (gpus.length === 0 && !value.gpuError) lines.push('no NVIDIA GPU reported');
  const processes = value.processes ?? [];
  if (processes.length > 0) {
    // Under WDDM nvidia-smi lists every GPU client — explorer, browsers, the
    // wallpaper engine — so a full dump buries the one process that matters.
    const hidden = processes.some((proc) => typeof proc.usedMiB !== 'number');
    const shown = processes.slice(0, 6);
    lines.push(`  GPU processes${hidden ? ' (WDDM reports no per-process VRAM, and graphics apps are mixed in)' : ''}:`);
    for (const proc of shown) {
      const size = typeof proc.usedMiB === 'number' ? `  ${proc.usedMiB} MiB` : '';
      lines.push(`    pid ${proc.pid}  ${proc.name}${size}`);
    }
    if (processes.length > shown.length) lines.push(`    … and ${processes.length - shown.length} more`);
  }
  lines.push('', 'CUDA-capable Python interpreters:');
  const interpreters = value.interpreters ?? [];
  if (interpreters.length === 0) lines.push('  (no candidates configured)');
  for (const item of interpreters) {
    if (!item.exists) {
      lines.push(`  - ${item.path}  [not found]`);
      continue;
    }
    if (item.error && !item.torch) {
      lines.push(`  - ${item.path}  python ${item.python ?? '?'}  no torch (${String(item.error).slice(0, 80)})`);
      continue;
    }
    const bits = [
      `python ${item.python ?? '?'}`,
      `torch ${item.torch ?? '?'}`,
      item.cuda ? `CUDA ok -> ${item.device} (${item.capability}, ${item.total_memory_MiB} MiB)` : 'CUDA UNAVAILABLE',
    ];
    if (item.cuda) bits.push(item.bf16 ? 'bf16 yes' : 'bf16 no');
    lines.push(`  - ${item.path}  ${bits.join('  ')}`);
  }
  lines.push('', `Recent jobs (${shortPath(runsDir, homeOf(runsDir))}):`);
  const jobs = value.jobs ?? [];
  if (jobs.length === 0) lines.push('  (no jobs yet)');
  for (const job of jobs) {
    const elapsed = job.status === 'running'
      ? formatMs(Date.now() - Date.parse(job.startedAt ?? Date.now()))
      : formatMs(job.elapsedMs ?? (job.finishedAt && job.startedAt
        ? Date.parse(job.finishedAt) - Date.parse(job.startedAt)
        : Number.NaN));
    const exit = job.exitCode === null || job.exitCode === undefined ? '' : ` exit=${job.exitCode}`;
    lines.push(`  ${job.id}${job.name ? ` "${job.name}"` : ''}  ${job.status}${exit}  ${elapsed}  (${job.mode})`);
  }
  return lines.join('\n');
}

/** The `~` root for a path that lives under the DSH home, if it does. */
function homeOf(path) {
  const marker = `${process.platform === 'win32' ? '\\' : '/'}.dsh`;
  const at = typeof path === 'string' ? path.indexOf(marker) : -1;
  return at > 0 ? path.slice(0, at) : undefined;
}

function renderJob(value, runsDir) {
  if (value.action === 'kill') {
    return value.killed
      ? `Killed job ${value.job} (pid ${value.pid}).`
      : `Job ${value.job} was not running (status ${value.status}).`;
  }
  const lines = [];
  const bits = [`job ${value.job}`];
  if (value.name) bits.push(`"${value.name}"`);
  if (value.status) bits.push(value.status);
  if (value.exitCode !== null && value.exitCode !== undefined) bits.push(`exit=${value.exitCode}`);
  if (value.elapsedMs) bits.push(formatMs(value.elapsedMs));
  if (value.stillRunning) bits.push('(still running)');
  if (value.alive === false && value.status === 'running') bits.push('(pid gone)');
  lines.push(bits.join('  '));
  if (value.python) lines.push(`python ${value.python}`);
  if (value.cwd) lines.push(`cwd    ${value.cwd}`);
  if (value.dir) lines.push(`dir    ${shortPath(value.dir, homeOf(runsDir ?? value.dir))}`);
  if (value.facts) {
    const gpuBits = [];
    if (value.facts.peak_alloc_MB !== undefined) {
      gpuBits.push(`peak VRAM ${value.facts.peak_alloc_MB} MB alloc / ${value.facts.peak_reserved_MB} MB reserved`);
    }
    if (value.facts.device) gpuBits.push(value.facts.device);
    if (gpuBits.length > 0) lines.push(`gpu    ${gpuBits.join('  ')}`);
  }
  if (value.note) lines.push(`note   ${value.note}`);
  if (Array.isArray(value.artifacts) && value.artifacts.length > 0) {
    lines.push(`files  ${value.artifacts.map((file) => `${file.name} (${formatBytes(file.bytes)})`).join(', ')}`);
  }
  const appendStream = (label, text) => {
    if (text === undefined) return;
    lines.push('', `--- ${label} ---`);
    lines.push(String(text).trim() === '' ? '(empty)' : String(text).replace(/\s+$/, ''));
  };
  appendStream('stdout', value.stdout);
  appendStream('stderr', value.stderr);
  if (value.stdoutTruncated) lines.push('', '[stdout truncated; the full log is on disk in the job directory]');
  return lines.join('\n');
}
