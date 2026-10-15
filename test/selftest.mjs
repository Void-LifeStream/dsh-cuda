#!/usr/bin/env node
/**
 * dsh-cuda engine self-test.
 *
 * Drives lib/engine.js directly with plain Node against the real GPU, so the
 * risky half of the plugin is verified without restarting the harness. Run:
 *
 *   & "$env:USERPROFILE\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe" test\selftest.mjs
 *
 * Options: --keep (leave the runs directory in place), --quick (skip GPU math).
 */
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { GpuEngine } from '../lib/engine.js';

const KEEP = process.argv.includes('--keep');
const QUICK = process.argv.includes('--quick');

let passed = 0;
let failed = 0;
const failures = [];

function check(label, condition, detail = '') {
  if (condition) {
    passed += 1;
    console.log(`  ok   ${label}${detail ? `  (${detail})` : ''}`);
  } else {
    failed += 1;
    failures.push(label);
    console.log(`  FAIL ${label}${detail ? `  (${detail})` : ''}`);
  }
}

function section(title) {
  console.log(`\n=== ${title} ===`);
}

const runsDir = await mkdtemp(join(tmpdir(), 'dsh-cuda-selftest-'));
console.log(`runs dir : ${runsDir}`);

const engine = new GpuEngine({
  runsDir,
  defaultTimeoutMs: 180_000,
  maxTimeoutMs: 300_000,
  maxOutputChars: 4_000,
});

try {
  /* ---------------------------------------------------------------- */
  section('interpreter discovery');
  const interpreters = await engine.interpreters({ refresh: true });
  for (const item of interpreters) {
    console.log(`  ${item.exists ? 'found' : 'absent'}  ${item.path}` +
      (item.torch ? `  python ${item.python} torch ${item.torch} cuda=${item.cuda}` : ''));
  }
  const usable = interpreters.filter((item) => item.cuda === true);
  check('at least one interpreter reports CUDA', usable.length > 0, `${usable.length} usable`);
  check('a resolved interpreter path exists', existsSync(await engine.resolveInterpreter()));

  /* ---------------------------------------------------------------- */
  section('device snapshot');
  const devices = await engine.devices();
  const gpu = devices.gpus[0];
  console.log(`  ${gpu?.name} driver ${gpu?.driver} ${gpu?.memoryUsedMiB}/${gpu?.memoryTotalMiB} MiB used, util ${gpu?.utilizationPct}%`);
  console.log(`  ${devices.processes.length} process(es) hold VRAM`);
  check('nvidia-smi reports a GPU', Boolean(gpu?.name));
  check('memory fields are numeric', Number(gpu?.memoryTotalMiB) > 0, `${gpu?.memoryTotalMiB} MiB`);

  /* ---------------------------------------------------------------- */
  if (!QUICK) {
    section('foreground GPU run');
    const math = await engine.runForeground({
      name: 'selftest-matmul',
      timeoutMs: 120_000,
      code: [
        'import os, time, torch',
        'assert torch.cuda.is_available(), "cuda unavailable"',
        'a = torch.randn(4096, 4096, device="cuda")',
        'b = torch.randn(4096, 4096, device="cuda")',
        'for _ in range(3):',
        '    c = a @ b',
        'torch.cuda.synchronize()',
        't0 = time.perf_counter()',
        'for _ in range(30):',
        '    c = a @ b',
        'torch.cuda.synchronize()',
        'dt = (time.perf_counter() - t0) / 30',
        'print(f"matmul 4096^3 {dt * 1e3:.2f} ms  {2 * 4096 ** 3 / dt / 1e12:.1f} TFLOPS fp32")',
        'print("device:", torch.cuda.get_device_name(0))',
        'job_dir = os.environ["DSH_CUDA_JOB_DIR"]',
        'with open(os.path.join(job_dir, "artifact.txt"), "w", encoding="utf-8") as fh:',
        '    fh.write(f"{dt:.6f}\\n")',
        'print("MARKER_FOREGROUND_OK")',
      ].join('\n'),
    });
    console.log(`  status ${math.status} exit ${math.exitCode} in ${math.elapsedMs} ms`);
    console.log(`  ${(math.stdout ?? '').trim().split('\n').join('\n  ')}`);
    check('foreground run finished with done', math.status === 'done', math.status);
    check('exit code is 0', math.exitCode === 0);
    check('stdout carries the marker', (math.stdout ?? '').includes('MARKER_FOREGROUND_OK'));
    check('wrapper recorded peak VRAM', Number(math.facts?.peak_alloc_MB) > 0, `${math.facts?.peak_alloc_MB} MB`);
    check('artifact written through DSH_CUDA_JOB_DIR', (math.artifacts ?? []).some((f) => f.name === 'artifact.txt'));

    section('failure reporting');
    const broken = await engine.runForeground({
      name: 'selftest-syntax-error',
      timeoutMs: 60_000,
      code: 'def broken(:\n    pass\n',
    });
    console.log(`  status ${broken.status} exit ${broken.exitCode}`);
    check('syntax error lands as failed', broken.status === 'failed', broken.status);
    check('non-zero exit code', broken.exitCode === 1, String(broken.exitCode));
    check('traceback reaches stderr', (broken.stderr ?? '').includes('SyntaxError'));

    section('runtime error reporting');
    const raised = await engine.runForeground({
      name: 'selftest-raise',
      timeoutMs: 60_000,
      code: 'print("before")\nraise ValueError("deliberate")\n',
    });
    check('uncaught exception lands as failed', raised.status === 'failed', raised.status);
    check('stdout before the raise is kept', (raised.stdout ?? '').includes('before'));
    check('stderr names the exception', (raised.stderr ?? '').includes('deliberate'));
  }

  /* ---------------------------------------------------------------- */
  section('background job');
  const background = await engine.runBackground({
    name: 'selftest-background',
    code: [
      'import time, sys, torch',
      'for index in range(3):',
      '    print(f"tick {index}", flush=True)',
      '    time.sleep(1)',
      'x = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")',
      'print("MARKER_BACKGROUND_OK", float(x[0, 0]))',
    ].join('\n'),
  });
  console.log(`  started ${background.id} pid ${background.pid}`);
  check('background run returns an id', /^\d{8}-\d{6}-/.test(background.id), background.id);
  const early = await engine.logs(background.id, { stream: 'stdout', tailChars: 2_000 });
  check('logs are readable while running', typeof early.stdout === 'string');
  const waited = await engine.wait(background.id, { timeoutMs: 120_000, pollMs: 500 });
  check('wait observes completion', waited.record.status === 'done', waited.record.status);
  const finished = await engine.describeJob(waited.dir, waited.record, { logLimit: 4_000 });
  console.log(`  ${(finished.stdout ?? '').trim().split('\n').join('\n  ')}`);
  check('background stdout is complete', (finished.stdout ?? '').includes('MARKER_BACKGROUND_OK'));
  check('background fact sheet exists', finished.facts?.ok === true);

  /* ---------------------------------------------------------------- */
  section('foreground budget promotes to background');
  const promoted = await engine.runForeground({
    name: 'selftest-promote',
    timeoutMs: 2_000,
    code: 'import time\ntime.sleep(15)\nprint("MARKER_PROMOTED_OK")\n',
  });
  console.log(`  status ${promoted.status}  note ${promoted.note ?? '(none)'}`);
  check('over-budget run is still running', promoted.status === 'running', promoted.status);
  check('promotion is explained', typeof promoted.note === 'string' && promoted.note.length > 0);
  const promotedDone = await engine.wait(promoted.job, { timeoutMs: 120_000, pollMs: 500 });
  check('promoted run completes on its own', promotedDone.record.status === 'done', promotedDone.record.status);

  /* ---------------------------------------------------------------- */
  section('cancellation');
  const victim = await engine.runBackground({
    name: 'selftest-kill',
    code: 'import time\nprint("sleeping", flush=True)\ntime.sleep(120)\nprint("should never print")\n',
  });
  await new Promise((resolveWait) => setTimeout(resolveWait, 1_500));
  const killed = await engine.kill(victim.id);
  console.log(`  ${JSON.stringify(killed)}`);
  check('kill reports success', killed.killed === true);
  await new Promise((resolveWait) => setTimeout(resolveWait, 1_000));
  const afterKill = await engine.status(victim.id);
  check('killed job is no longer alive', afterKill.alive === false, String(afterKill.alive));
  check('record says killed', afterKill.record.status === 'killed', afterKill.record.status);

  section('job listing and resolution');
  const jobs = await engine.listJobs({ limit: 10 });
  console.log(`  ${jobs.length} job(s): ${jobs.map((job) => `${job.id}:${job.status}`).join(' ')}`);
  check('jobs are listed newest first', jobs.length >= 4 && jobs[0].id >= jobs[jobs.length - 1].id);
  const latest = await engine.resolveJobId('latest');
  check('`latest` resolves', typeof latest === 'string' && latest.length > 0, latest);
  const prefix = await engine.resolveJobId(background.id.slice(0, 15));
  check('id prefix resolves', prefix === background.id, prefix);

  section('input validation');
  let threw = '';
  try {
    await engine.runForeground({ code: 'print(1)', script: 'x.py' });
  } catch (error) {
    threw = String(error.message);
  }
  check('code+script together is rejected', threw.includes('not both'), threw.slice(0, 60));
  threw = '';
  try {
    await engine.runForeground({});
  } catch (error) {
    threw = String(error.message);
  }
  check('empty submission is rejected', threw.includes('required'), threw.slice(0, 60));
  threw = '';
  try {
    await engine.status('19990101-000000-zzzz');
  } catch (error) {
    threw = String(error.message);
  }
  check('unknown job id is rejected', threw.includes('unknown job'), threw.slice(0, 60));

  section('script-file mode');
  const scriptPath = join(runsDir, 'external.py');
  await writeFile(scriptPath, 'import sys\nprint("argv:", sys.argv[1:])\nprint("MARKER_SCRIPT_OK")\n', 'utf8');
  const fromFile = await engine.runForeground({
    script: scriptPath,
    args: ['--alpha', '0.5', 'plain'],
    name: 'selftest-script',
    timeoutMs: 60_000,
  });
  check('script file mode works', (fromFile.stdout ?? '').includes('MARKER_SCRIPT_OK'), fromFile.status);
  check('args reach the program as sys.argv[1:]', (fromFile.stdout ?? '').includes("['--alpha', '0.5', 'plain']"),
    (fromFile.stdout ?? '').split('\n')[0]);
  check('the run copied the source', existsSync(join(fromFile.dir, 'script.py')));
  const copied = await readFile(join(fromFile.dir, 'script.py'), 'utf8');
  check('the copy matches the original', copied.includes('MARKER_SCRIPT_OK'));
} finally {
  if (KEEP) {
    console.log(`\nruns kept at ${runsDir}`);
  } else {
    await rm(runsDir, { recursive: true, force: true }).catch(() => {});
  }
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log('failed checks: ' + failures.join('; '));
  process.exit(1);
}
