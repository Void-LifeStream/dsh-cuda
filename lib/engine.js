/**
 * dsh-cuda — GPU compute engine (pure Node, no DSH imports).
 *
 * Everything a DSH tool needs to run Python work on the local CUDA device:
 * interpreter discovery, device snapshots, foreground and background runs,
 * durable job records, log tails, and cancellation.
 *
 * Kept free of DSH imports on purpose: `test/selftest.mjs` drives this module
 * directly with the bundled Node, so the risky half of the plugin is verified
 * without restarting the harness.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, open, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { isAbsolute, join, resolve } from 'node:path';

/* ------------------------------------------------------------------ *
 * Constants and small helpers
 * ------------------------------------------------------------------ */

/** Executable names worth trying, in order, when nothing else is configured. */
const PYTHON_NAMES = process.platform === 'win32'
  ? ['python.exe', 'python3.exe']
  : ['python3', 'python'];

/**
 * Interpreters to probe when the caller configured none: everything named
 * `python` on PATH, then the places a GPU box usually hides one. Deliberately
 * free of machine-specific paths — point `DSH_CUDA_PYTHON` or `~/.dsh/cuda.json`
 * at an embedded interpreter (ComfyUI's, a conda env) when it lives elsewhere.
 * @returns {string[]} existing candidate paths, de-duplicated.
 */
export function discoverInterpreters() {
  const found = [];
  const separator = process.platform === 'win32' ? ';' : ':';
  for (const dir of (process.env.PATH ?? '').split(separator)) {
    if (dir === '') continue;
    for (const name of PYTHON_NAMES) {
      const candidate = join(dir, name);
      if (existsSync(candidate)) found.push(candidate);
    }
  }
  const guesses = process.platform === 'win32'
    ? [
        join(homedir(), 'miniconda3', 'python.exe'),
        join(homedir(), 'anaconda3', 'python.exe'),
        join(homedir(), 'AppData', 'Local', 'Programs', 'Python', 'Python313', 'python.exe'),
        join(homedir(), 'AppData', 'Local', 'Programs', 'Python', 'Python312', 'python.exe'),
        join(homedir(), 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'python.exe'),
      ]
    : ['/usr/bin/python3', '/usr/local/bin/python3'];
  for (const guess of guesses) if (existsSync(guess)) found.push(guess);
  return [...new Set(found)];
}

/**
 * Optional local settings, so a deployment can point at an embedded Python
 * without touching its own environment: `~/.dsh/cuda.json` with
 * `{"pythonPath": "...", "pythonCandidates": ["..."], "runsDir": "..."}`.
 * A malformed file is ignored on purpose — the plugin must still start.
 * @returns {object} parsed settings, or an empty object.
 */
export function readLocalSettings() {
  try {
    const parsed = JSON.parse(readFileSync(join(dshHome(), 'cuda.json'), 'utf8'));
    return parsed !== null && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

/** Files a run directory owns; everything else counts as a produced artifact. */
const OWNED_FILES = new Set(['script.py', '_runner.py', 'stdout.log', 'stderr.log', 'job.json', 'result.json']);

/** Wrapper that runs the user's script and records facts; written per job. */
const RUNNER_SOURCE = `"""dsh-cuda run wrapper: exec the target script and record resource facts."""
import json, runpy, sys, time, traceback

_script, _result = sys.argv[1], sys.argv[2]
sys.argv = [_script] + sys.argv[3:]


def _gpu_facts():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "device": torch.cuda.get_device_name(0),
                "peak_alloc_MB": round(torch.cuda.max_memory_allocated() / 1048576, 1),
                "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / 1048576, 1),
                "torch": torch.__version__,
            }
    except Exception:
        pass
    return {}


_started = time.perf_counter()
_ok, _code, _error = True, 0, None
try:
    runpy.run_path(_script, run_name="__main__")
except SystemExit as _exc:
    _code = _exc.code if isinstance(_exc.code, int) else (0 if _exc.code is None else 1)
    _ok = _code == 0
    if not _ok:
        _error = "SystemExit(%r)" % (_exc.code,)
except BaseException:
    _ok, _code, _error = False, 1, "uncaught exception"
    traceback.print_exc()

_facts = {
    "ok": _ok,
    "exit_code": _code,
    "elapsed_s": round(time.perf_counter() - _started, 3),
    "error": _error,
    "pid": __import__("os").getpid(),
}
_facts.update(_gpu_facts())
try:
    with open(_result, "w", encoding="utf-8") as _fh:
        json.dump(_facts, _fh, ensure_ascii=False)
except Exception:
    pass
sys.exit(_code)
`;

/** Template for the one-shot interpreter probe. */
const PROBE_SOURCE = `import json, sys
info = {"python": sys.version.split()[0], "executable": sys.executable, "torch": None,
        "cuda": False, "device": None, "capability": None, "total_memory_MiB": None,
        "bf16": False, "capable": [], "error": None}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info["device"] = p.name
        info["capability"] = "sm_%d%d" % (p.major, p.minor)
        info["total_memory_MiB"] = round(p.total_memory / 1048576)
        info["bf16"] = bool(torch.cuda.is_bf16_supported())
    info["capable"] = list(torch.cuda.get_arch_list())
except Exception as exc:
    info["error"] = type(exc).__name__ + ": " + str(exc)[:200]
print(json.dumps(info))
`;

export function dshHome() {
  return process.env.DSH_HOME || join(homedir(), '.dsh');
}

/** A positive number from the environment, or the fallback. */
function envNumber(name, fallback) {
  const raw = Number(process.env[name]);
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}

export function newJobId(date = new Date()) {
  const pad = (value, width = 2) => String(value).padStart(width, '0');
  const stamp =
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}` +
    `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
  return `${stamp}-${Math.random().toString(36).slice(2, 6)}`;
}

/** Read at most `maxBytes` from the end of a file; '' when it does not exist. */
export async function readTail(path, maxBytes = 16_384) {
  let handle;
  try {
    handle = await open(path, 'r');
  } catch {
    return '';
  }
  try {
    const info = await handle.stat();
    const length = Math.min(maxBytes, info.size);
    if (length === 0) return '';
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, info.size - length);
    const text = buffer.toString('utf8');
    // A tail may start mid-character; drop the first partial line.
    const cut = info.size > length ? text.indexOf('\n') : -1;
    return cut === -1 ? text : text.slice(cut + 1);
  } catch {
    return '';
  } finally {
    await handle.close().catch(() => {});
  }
}

function tailText(text, limit) {
  if (text.length <= limit) return text;
  return `[... ${text.length - limit} earlier characters omitted ...]\n` + text.slice(-limit);
}

function isAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === 'EPERM';
  }
}

/* ------------------------------------------------------------------ *
 * Engine
 * ------------------------------------------------------------------ */

export class GpuEngine {
  /**
   * @param {object} config resolved plugin configuration.
   */
  constructor(config = {}) {
    this.config = config;
    const local = readLocalSettings();
    // Priority: explicit config > environment > ~/.dsh/cuda.json > discovery.
    this.pythonPath = config.pythonPath?.trim()
      || process.env.DSH_CUDA_PYTHON?.trim()
      || String(local.pythonPath ?? '').trim()
      || '';
    const configured = config.pythonCandidates?.length
      ? config.pythonCandidates
      : (Array.isArray(local.pythonCandidates) ? local.pythonCandidates : []);
    this.candidates = (configured.length > 0 ? configured : discoverInterpreters())
      .map((entry) => String(entry).trim())
      .filter(Boolean);
    this.runsDir = config.runsDir?.trim()
      || process.env.DSH_CUDA_RUNS_DIR?.trim()
      || String(local.runsDir ?? '').trim()
      || join(dshHome(), 'gpu-runs');
    // Environment variables stand in for plugin Config: this package installs as
    // a `link:` dependency outside the profile tree and therefore cannot import
    // `@deepseek-ai/schemastery` to declare a Config schema.
    this.defaultTimeoutMs = config.defaultTimeoutMs ?? envNumber('DSH_CUDA_TIMEOUT_MS', 300_000);
    this.maxTimeoutMs = config.maxTimeoutMs ?? envNumber('DSH_CUDA_MAX_TIMEOUT_MS', 1_800_000);
    this.maxOutputChars = config.maxOutputChars ?? envNumber('DSH_CUDA_MAX_OUTPUT_CHARS', 12_000);
    this.extraEnv = config.env ?? {};
    /** @type {Map<string, {at: number, info: object}>} */
    this.probeCache = new Map();
  }

  /** Interpreters to probe: the configured one first, then the defaults. */
  interpreterCandidates() {
    const list = [];
    if (this.pythonPath) list.push(this.pythonPath);
    for (const entry of this.candidates) if (!list.includes(entry)) list.push(entry);
    const fromEnv = process.env.DSH_CUDA_PYTHON;
    if (fromEnv && !list.includes(fromEnv)) list.unshift(fromEnv);
    return list;
  }

  /** The interpreter a run should use: the first candidate that exists. */
  async resolveInterpreter() {
    if (this.pythonPath) {
      if (!existsSync(this.pythonPath)) {
        throw new Error(
          `configured pythonPath does not exist: ${this.pythonPath}. ` +
            'Fix the dsh-cuda config or leave it empty to auto-detect.',
        );
      }
      return this.pythonPath;
    }
    for (const candidate of this.interpreterCandidates()) {
      if (existsSync(candidate)) return candidate;
    }
    throw new Error(
      'no CUDA-capable Python interpreter found. Probed: ' +
        `${this.interpreterCandidates().join(', ') || '(none configured)'}. ` +
        'Set DSH_CUDA_PYTHON in the environment (or pythonPath in the plugin config, when Config is available).',
    );
  }

  /**
   * Probe every candidate interpreter once (cached for `ttlMs`) and report what
   * each can do. Torch import costs seconds, so the cache matters.
   */
  async interpreters({ refresh = false, ttlMs = 3_600_000, signal } = {}) {
    const results = [];
    for (const path of this.interpreterCandidates()) {
      if (!existsSync(path)) {
        results.push({ path, exists: false, cuda: false, note: 'not found' });
        continue;
      }
      const cached = this.probeCache.get(path);
      if (!refresh && cached && Date.now() - cached.at < ttlMs) {
        results.push({ path, exists: true, ...cached.info, cached: true });
        continue;
      }
      try {
        const info = await this.#probeInterpreter(path, signal);
        this.probeCache.set(path, { at: Date.now(), info });
        results.push({ path, exists: true, ...info });
      } catch (error) {
        results.push({ path, exists: true, cuda: false, error: String(error?.message ?? error) });
      }
    }
    return results;
  }

  async #probeInterpreter(path, signal) {
    const { stdout } = await this.spawnCapture(path, ['-c', PROBE_SOURCE], {
      timeoutMs: 120_000,
      signal,
      cwd: this.runsDir,
    });
    const line = stdout.trim().split('\n').filter(Boolean).pop() ?? '';
    try {
      return JSON.parse(line);
    } catch {
      throw new Error(`probe of ${path} returned unparsable output: ${line.slice(0, 200)}`);
    }
  }

  /** One nvidia-smi query, parsed into plain objects. */
  async devices({ signal } = {}) {
    const gpu = await this.#nvidiaSmi(
      'name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,compute_cap',
      { signal },
    );
    const names = [
      'name', 'driver', 'memoryTotalMiB', 'memoryUsedMiB', 'memoryFreeMiB',
      'utilizationPct', 'temperatureC', 'powerW', 'computeCapability',
    ];
    const gpus = gpu.map((row) => {
      const record = {};
      names.forEach((key, index) => { record[key] = row[index] ?? ''; });
      return record;
    });
    const apps = await this.#nvidiaSmi('pid,process_name,used_memory', { signal, computeApps: true });
    const processes = apps.map((row) => ({
      pid: Number(row[0]),
      // Full paths are noise here; two pythons stay distinguishable by pid.
      name: String(row[1] ?? '').split(/[\\/]/).pop() ?? '',
      // Under WDDM nvidia-smi reports [N/A] instead of a per-process figure.
      usedMiB: /^\d+$/.test(row[2] ?? '') ? Number(row[2]) : null,
    }));
    return { gpus, processes };
  }

  async #nvidiaSmi(query, { signal, computeApps = false } = {}) {
    const prefix = computeApps ? '--query-compute-apps=' : '--query-gpu=';
    const exe = process.env.ComSpec ? 'nvidia-smi.exe' : 'nvidia-smi';
    const { stdout, code } = await this.spawnCapture(
      exe,
      [prefix + query, '--format=csv,noheader,nounits'],
      { timeoutMs: 30_000, signal },
    ).catch((error) => ({ stdout: '', stderr: String(error?.message ?? error), code: -1 }));
    if (code !== 0 && stdout.trim() === '') return [];
    return stdout
      .trim()
      .split('\n')
      .map((line) => line.split(',').map((cell) => cell.trim()))
      .filter((row) => row.length > 0 && row[0] !== '');
  }

  /* ---------------------------------------------------------------- *
   * Running work
   * ---------------------------------------------------------------- */

  /**
   * Prepare a job directory: write the script, the wrapper, and the record.
   * @returns {Promise<{id: string, dir: string, scriptPath: string, runnerPath: string, record: object}>}
   */
  async #prepare({ code, script, name, cwd, timeoutMs, mode, env, args }) {
    if ((code === undefined || code === null || String(code).trim() === '') && !script) {
      throw new Error('either `code` (Python source) or `script` (path to a .py file) is required');
    }
    if (code !== undefined && code !== null && script) {
      throw new Error('pass either `code` or `script`, not both');
    }
    const python = await this.resolveInterpreter();
    const id = newJobId();
    const dir = join(this.runsDir, id);
    await mkdir(dir, { recursive: true });

    const scriptPath = join(dir, 'script.py');
    if (script) {
      const source = isAbsolute(script) ? script : resolve(cwd ?? process.cwd(), script);
      const info = await stat(source).catch(() => undefined);
      if (!info?.isFile()) throw new Error(`script not found: ${source}`);
      // Copy so the run stays reproducible after the original is edited.
      await writeFile(scriptPath, await readFile(source, 'utf8'), 'utf8');
    } else {
      await writeFile(scriptPath, String(code), 'utf8');
    }
    const runnerPath = join(dir, '_runner.py');
    await writeFile(runnerPath, RUNNER_SOURCE, 'utf8');
    await writeFile(join(dir, 'stdout.log'), '', 'utf8');
    await writeFile(join(dir, 'stderr.log'), '', 'utf8');

    const record = {
      id,
      name: name?.trim() || null,
      mode,
      status: 'starting',
      python,
      cwd: cwd ?? process.cwd(),
      dir,
      scriptPath,
      startedAt: new Date().toISOString(),
      finishedAt: null,
      exitCode: null,
      pid: null,
      timeoutMs: timeoutMs ?? null,
      args: (args ?? []).map((entry) => String(entry)),
      env: Object.keys({ ...this.extraEnv, ...(env ?? {}) }),
    };
    await this.#writeRecord(dir, record);
    return { id, dir, scriptPath, runnerPath, record };
  }

  async #writeRecord(dir, record) {
    await writeFile(join(dir, 'job.json'), JSON.stringify(record, null, 2), 'utf8');
  }

  /** Environment for every child: UTF-8 IO and unbuffered logs are mandatory. */
  #childEnv(extra) {
    return {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
      PYTHONUTF8: '1',
      ...this.extraEnv,
      ...(extra ?? {}),
    };
  }

  /**
   * Run a script to completion (or timeout) and return everything the model
   * needs to judge the outcome.
   */
  async runForeground({ code, script, cwd, name, timeoutMs, env, signal, onProgress, args } = {}) {
    const effectiveTimeout = Math.min(
      Math.max(timeoutMs ?? this.defaultTimeoutMs, 1_000),
      this.maxTimeoutMs,
    );
    const job = await this.#prepare({ code, script, name, cwd, timeoutMs: effectiveTimeout, mode: 'foreground', env, args });
    const stdoutPath = join(job.dir, 'stdout.log');
    const stderrPath = join(job.dir, 'stderr.log');
    const out = await open(stdoutPath, 'a');
    const err = await open(stderrPath, 'a');

    const child = spawn(job.record.python, [job.runnerPath, job.scriptPath, join(job.dir, 'result.json'), ...(job.record.args ?? [])], {
      cwd: job.record.cwd,
      env: this.#childEnv({ DSH_CUDA_JOB_DIR: job.dir, ...(env ?? {}) }),
      windowsHide: true,
      stdio: ['ignore', out.fd, err.fd],
    });
    job.record.pid = child.pid;
    job.record.status = 'running';
    await this.#writeRecord(job.dir, job.record);

    const startedAt = Date.now();
    let timedOut = false;
    let aborted = false;
    const timer = setTimeout(() => { timedOut = true; }, effectiveTimeout);
    const onAbort = () => { aborted = true; };
    signal?.addEventListener('abort', onAbort, { once: true });

    const exit = await new Promise((resolveExit) => {
      let settled = false;
      let watchdog;
      const settle = (value) => {
        if (settled) return;
        settled = true;
        clearInterval(watchdog);
        resolveExit(value);
      };
      child.once('error', (error) => settle({ code: null, error }));
      child.once('exit', (code, sig) => settle({ code, sig }));
      watchdog = setInterval(() => {
        // Exceeding the foreground budget promotes the run to background work: a
        // multi-hour job must not be destroyed just because the caller stopped
        // waiting. An explicit cancel is different - that one really ends it.
        if (timedOut) settle({ code: null, sig: null, detached: true });
        else if (aborted) {
          this.killTree(child.pid);
          settle({ code: null, sig: 'SIGKILL', aborted: true });
        }
      }, 250);
      if (onProgress) onProgress({ pid: child.pid });
    });

    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
    await out.close().catch(() => {});
    await err.close().catch(() => {});

    if (exit.detached) {
      job.record.status = 'running';
      job.record.detachedAt = new Date().toISOString();
      job.record.promoted = true;
      job.record.note = 'foreground budget exhausted; still running in the background — follow it with cuda_job instead of starting it again';
    } else if (exit.aborted) {
      job.record.status = 'killed';
      job.record.finishedAt = new Date().toISOString();
    } else {
      job.record.status = exit.code === 0 ? 'done' : 'failed';
    }
    job.record.exitCode = exit.code ?? null;
    job.record.signal = exit.sig ?? null;
    if (!exit.detached) job.record.finishedAt = job.record.finishedAt ?? new Date().toISOString();
    job.record.elapsedMs = Date.now() - startedAt;
    if (exit.error) job.record.spawnError = String(exit.error.message ?? exit.error);
    await this.#writeRecord(job.dir, job.record);

    return this.describeJob(job.dir, job.record, { logLimit: this.maxOutputChars });
  }

  /**
   * Start a detached run and return immediately. The child survives a harness
   * restart on purpose: multi-hour science should not die with the UI.
   */
  async runBackground({ code, script, cwd, name, env, args } = {}) {
    const job = await this.#prepare({ code, script, name, cwd, mode: 'background', env, args });
    const out = await open(join(job.dir, 'stdout.log'), 'a');
    const err = await open(join(job.dir, 'stderr.log'), 'a');
    const child = spawn(job.record.python, [job.runnerPath, job.scriptPath, join(job.dir, 'result.json'), ...(job.record.args ?? [])], {
      cwd: job.record.cwd,
      env: this.#childEnv({ DSH_CUDA_JOB_DIR: job.dir, ...(env ?? {}) }),
      windowsHide: true,
      detached: true,
      stdio: ['ignore', out.fd, err.fd],
    });
    child.unref();
    await out.close().catch(() => {});
    await err.close().catch(() => {});
    job.record.pid = child.pid;
    job.record.status = 'running';
    await this.#writeRecord(job.dir, job.record);
    return { id: job.id, pid: child.pid, dir: job.dir, mode: 'background' };
  }

  /** Kill a process tree (a Python run may own worker children). */
  killTree(pid) {
    if (!Number.isInteger(pid) || pid <= 0) return false;
    try {
      if (process.platform === 'win32') {
        spawn('taskkill.exe', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
      } else {
        process.kill(pid, 'SIGKILL');
      }
      return true;
    } catch {
      return false;
    }
  }

  /* ---------------------------------------------------------------- *
   * Job records
   * ---------------------------------------------------------------- */

  jobDir(id) {
    const safe = String(id ?? '').replace(/[^0-9a-zA-Z-]/g, '');
    if (safe === '') throw new Error('job id is required');
    return join(this.runsDir, safe);
  }

  async #loadRecord(dir) {
    try {
      return JSON.parse(await readFile(join(dir, 'job.json'), 'utf8'));
    } catch {
      return undefined;
    }
  }

  /**
   * A detached run has no parent left to record its exit, so the wrapper's own
   * result.json is the truth. Settle such a record once, then persist it.
   */
  async #reconcile(dir, record) {
    if (record.status !== 'running') return record;
    if (isAlive(record.pid)) return record;
    const facts = await this.#resultFacts(dir);
    if (facts) {
      record.status = facts.ok ? 'done' : 'failed';
      record.exitCode = facts.exit_code ?? record.exitCode ?? null;
      if (typeof facts.elapsed_s === 'number') record.elapsedMs = Math.round(facts.elapsed_s * 1000);
    } else {
      // No fact sheet: the process died before the wrapper could write one.
      record.status = 'lost';
    }
    record.finishedAt = record.finishedAt ?? new Date().toISOString();
    delete record.note;
    await this.#writeRecord(dir, record).catch(() => {});
    return record;
  }

  async readJob(id) {
    const jobId = await this.resolveJobId(id);
    const dir = join(this.runsDir, jobId);
    const record = await this.#loadRecord(dir);
    if (!record) throw new Error(`unknown job "${id}" (no job.json under ${this.runsDir})`);
    return { id: jobId, dir, record };
  }

  /** Resolve `latest`, an exact job id, or an unambiguous id prefix. */
  async resolveJobId(id) {
    if (id === undefined || id === null || String(id).trim() === '' || id === 'latest') {
      const jobs = await this.listJobs({ limit: 1 });
      if (jobs.length === 0) throw new Error(`no jobs yet under ${this.runsDir}`);
      return jobs[0].id;
    }
    const wanted = String(id).trim().replace(/[^0-9a-zA-Z-]/g, '');
    if (wanted === '') throw new Error('job id is required');
    if (existsSync(join(this.runsDir, wanted))) return wanted;
    const entries = await readdir(this.runsDir, { withFileTypes: true }).catch(() => []);
    const matches = entries
      .filter((entry) => entry.isDirectory() && entry.name.startsWith(wanted))
      .map((entry) => entry.name);
    if (matches.length === 1) return matches[0];
    if (matches.length > 1) throw new Error(`job id "${wanted}" is ambiguous: ${matches.slice(0, 6).join(', ')}`);
    throw new Error(`unknown job "${wanted}": nothing under ${this.runsDir} matches`);
  }

  /** Live status: a record whose pid still exists is running, whoever started it. */
  async status(id) {
    const jobId = await this.resolveJobId(id);
    const dir = join(this.runsDir, jobId);
    const record = await this.#loadRecord(dir);
    if (!record) throw new Error(`unknown job "${id}" (no job.json under ${this.runsDir})`);
    await this.#reconcile(dir, record);
    return { dir, record, alive: isAlive(record.pid) };
  }

  async listJobs({ limit = 10 } = {}) {
    const entries = await readdir(this.runsDir, { withFileTypes: true }).catch(() => []);
    const dirs = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort().reverse();
    const jobs = [];
    for (const name of dirs.slice(0, limit)) {
      const dir = join(this.runsDir, name);
      const record = await this.#loadRecord(dir);
      if (!record) continue;
      await this.#reconcile(dir, record);
      jobs.push({
        id: record.id ?? name,
        name: record.name ?? null,
        mode: record.mode ?? 'foreground',
        status: record.status,
        pid: record.pid ?? null,
        startedAt: record.startedAt ?? null,
        finishedAt: record.finishedAt ?? null,
        exitCode: record.exitCode ?? null,
        elapsedMs: record.elapsedMs ?? null,
      });
    }
    return jobs;
  }

  /** Everything worth reporting about one finished or running job. */
  async describeJob(dir, record, { logLimit = 8_000 } = {}) {
    await this.#reconcile(dir, record);
    const [stdout, stderr, artifacts, facts] = await Promise.all([
      readTail(join(dir, 'stdout.log'), 512 * 1024),
      readTail(join(dir, 'stderr.log'), 256 * 1024),
      this.#artifacts(dir),
      this.#resultFacts(dir),
    ]);
    return {
      job: record.id,
      name: record.name ?? null,
      mode: record.mode ?? 'foreground',
      status: record.status,
      pid: record.pid ?? null,
      python: record.python ?? null,
      cwd: record.cwd ?? null,
      dir,
      exitCode: record.exitCode ?? null,
      elapsedMs: record.elapsedMs ?? (record.startedAt ? Date.now() - Date.parse(record.startedAt) : null),
      note: record.note ?? (record.promoted === true ? 'promoted: it outlived its foreground budget and then finished on its own' : undefined),
      facts: facts ?? null,
      artifacts,
      stdout: tailText(stdout, logLimit),
      stderr: tailText(stderr, Math.min(4_000, Math.max(1_000, Math.floor(logLimit / 3)))),
      stdoutTruncated: stdout.length > logLimit,
    };
  }

  async #resultFacts(dir) {
    try {
      return JSON.parse(await readFile(join(dir, 'result.json'), 'utf8'));
    } catch {
      return null;
    }
  }

  async #artifacts(dir) {
    const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
    const files = [];
    for (const entry of entries) {
      if (!entry.isFile() || OWNED_FILES.has(entry.name)) continue;
      const info = await stat(join(dir, entry.name)).catch(() => undefined);
      if (info) files.push({ name: entry.name, bytes: info.size });
    }
    return files.sort((a, b) => a.name.localeCompare(b.name));
  }

  /** Log tail for one job, by stream. */
  async logs(id, { stream = 'both', tailChars = 8_000 } = {}) {
    const { dir, record } = await this.readJob(id);
    const result = { job: record.id, status: record.status, dir };
    if (stream === 'stdout' || stream === 'both') {
      result.stdout = tailText(await readTail(join(dir, 'stdout.log'), 512 * 1024), tailChars);
    }
    if (stream === 'stderr' || stream === 'both') {
      result.stderr = tailText(await readTail(join(dir, 'stderr.log'), 256 * 1024), tailChars);
    }
    return result;
  }

  /** Block until the job's pid disappears, or the wait budget runs out. */
  async wait(id, { timeoutMs = 600_000, pollMs = 1_000, signal } = {}) {
    const jobId = await this.resolveJobId(id);
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const { record } = await this.readJob(jobId);
      const alive = isAlive(record.pid);
      if (!alive) return this.status(jobId);
      if (Date.now() >= deadline) return { ...(await this.status(jobId)), waited: true, stillRunning: true };
      if (signal?.aborted) return { ...(await this.status(jobId)), aborted: true, stillRunning: true };
      await new Promise((resolveWait) => setTimeout(resolveWait, Math.min(pollMs, 1_000)));
    }
  }

  /** Terminate a running job and mark the record. */
  async kill(id) {
    const jobId = await this.resolveJobId(id);
    const { dir, record } = await this.readJob(jobId);
    const alive = isAlive(record.pid);
    if (!alive) return { job: jobId, killed: false, status: record.status, note: 'process is not running' };
    const killed = this.killTree(record.pid);
    record.status = 'killed';
    record.finishedAt = new Date().toISOString();
    await this.#writeRecord(dir, record).catch(() => {});
    return { job: jobId, killed, status: 'killed', pid: record.pid };
  }

  /* ---------------------------------------------------------------- *
   * Low-level capture (probes and nvidia-smi only)
   * ---------------------------------------------------------------- */

  spawnCapture(exe, args, { timeoutMs = 60_000, signal, cwd } = {}) {
    return new Promise((resolveCapture, reject) => {
      const child = spawn(exe, args, {
        cwd: cwd && existsSync(cwd) ? cwd : undefined,
        env: this.#childEnv(),
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        this.killTree(child.pid);
        reject(new Error(`command timed out after ${timeoutMs} ms: ${exe}`));
      }, timeoutMs);
      const onAbort = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.killTree(child.pid);
        reject(new Error('aborted'));
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      child.stdout.on('data', (chunk) => { stdout = tailText(stdout + chunk.toString('utf8'), 1_000_000); });
      child.stderr.on('data', (chunk) => { stderr = tailText(stderr + chunk.toString('utf8'), 200_000); });
      child.once('error', (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        reject(error);
      });
      child.once('close', (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        resolveCapture({ stdout, stderr, code });
      });
    });
  }
}
