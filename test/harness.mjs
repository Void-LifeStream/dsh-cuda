#!/usr/bin/env node
/**
 * dsh-cuda plugin harness test.
 *
 * The harness only loads bundle code at startup, so `apply()` cannot be
 * observed from a running session. This drives it the other way: a fake cordis
 * context with the exact surface the plugin uses, then the real tools are
 * invoked end to end (including a genuine GPU run).
 *
 *   & "$env:USERPROFILE\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe" test\harness.mjs
 */
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { apply, name, inject } from '../lib/entry.js';

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

/* ------------------------------------------------------------------ *
 * A minimal cordis context: only what the plugin actually touches.
 * ------------------------------------------------------------------ */

const registered = new Map();
const effects = [];
let seenSkillProvider = null;
let injectSeen = null;

const skillsCtx = {
  effect(fn, label) {
    const dispose = fn();
    effects.push({ label, dispose });
    return dispose;
  },
  skills: {
    registerProvider(create) {
      seenSkillProvider = create({ signal: new AbortController().signal, invalidate() {} });
      return () => { seenSkillProvider = null; };
    },
  },
  logger: { warn: (...parts) => console.log('  [warn]', ...parts) },
};

const ctx = {
  effect(fn, label) {
    const dispose = fn();
    effects.push({ label, dispose });
    return dispose;
  },
  tools: {
    register(definition) {
      if (typeof definition?.name !== 'string' || definition.name === '') {
        throw new Error('tool definition needs a name');
      }
      if (registered.has(definition.name)) throw new Error(`duplicate tool ${definition.name}`);
      registered.set(definition.name, definition);
      return () => registered.delete(definition.name);
    },
  },
  inject(deps, callback) {
    injectSeen = deps;
    callback(skillsCtx);
  },
  logger: { warn: (...parts) => console.log('  [warn]', ...parts) },
};

const runsDir = await mkdtemp(join(tmpdir(), 'dsh-cuda-harness-'));
console.log(`runs dir : ${runsDir}`);

try {
  /* ---------------------------------------------------------------- */
  console.log('\n=== plugin surface ===');
  check('plugin name is dsh-cuda', name === 'dsh-cuda', name);
  check('tools is a hard dependency', Array.isArray(inject) && inject.includes('tools'), JSON.stringify(inject));

  /* ---------------------------------------------------------------- */
  console.log('\n=== apply() ===');
  apply(ctx, { runsDir });
  check('skills is requested softly', injectSeen !== null && injectSeen.includes('skills'), JSON.stringify(injectSeen));
  check('apply registered three tools', registered.size === 3, [...registered.keys()].join(', '));
  for (const toolName of ['cuda_status', 'cuda_run', 'cuda_job']) {
    check(`${toolName} is registered`, registered.has(toolName));
  }
  check('every registration is owned by an effect', effects.length === 4, `${effects.length} effects`);

  /* ---------------------------------------------------------------- */
  console.log('\n=== tool definitions ===');
  for (const [toolName, definition] of registered) {
    const parameters = definition.parameters;
    const okShape =
      parameters?.type === 'object' &&
      parameters.properties !== null &&
      typeof parameters.properties === 'object' &&
      typeof definition.output?.schema === 'object' &&
      typeof definition.output?.render === 'function' &&
      typeof definition.execute === 'function' &&
      typeof definition.description === 'string' &&
      definition.description.length > 40;
    check(`${toolName} has a well-formed definition`, okShape);
    check(`${toolName} declares object parameters`, parameters?.type === 'object');
    const props = Object.keys(parameters?.properties ?? {});
    check(`${toolName} declares properties`, props.length > 0, props.join(', '));
    // Required entries must name declared properties.
    for (const required of parameters?.required ?? []) {
      check(`${toolName}.${required} is required and declared`, props.includes(required));
    }
  }
  check('cuda_job requires its job argument', (registered.get('cuda_job').parameters.required ?? []).includes('job'));
  check('cuda_run exposes background', 'background' in registered.get('cuda_run').parameters.properties);
  check('cuda_status is concurrency safe', registered.get('cuda_status').isConcurrencySafe?.() === true);
  check('cuda_run is not concurrency safe', registered.get('cuda_run').isConcurrencySafe?.() !== true);

  /* ---------------------------------------------------------------- */
  console.log('\n=== skill provider ===');
  check('a skill provider was registered', seenSkillProvider !== null);
  if (seenSkillProvider) {
    check('provider is named dsh-cuda', seenSkillProvider.name === 'dsh-cuda', seenSkillProvider.name);
    const candidates = await seenSkillProvider.list({});
    check('provider lists one skill', candidates.length === 1, String(candidates.length));
    const candidate = candidates[0];
    check('candidate is cuda-compute', candidate?.name === 'cuda-compute', candidate?.name);
    check('candidate is model-invocable', candidate?.invocation?.modelInvocable === true);
    check('candidate carries a directory resource base', candidate?.resourceBase?.kind === 'directory', candidate?.resourceBase?.path);
    const skill = await seenSkillProvider.get(candidate, {});
    check('skill body loads', typeof skill?.content === 'string' && skill.content.length > 1000, `${skill?.content?.length} chars`);
    check('frontmatter is stripped', !skill.content.startsWith('---'));
    check('body documents the tools', skill.content.includes('cuda_run') && skill.content.includes('cuda_status'));
    check('body names the interpreter', skill.content.includes('ComfyUI'));
  }

  /* ---------------------------------------------------------------- */
  console.log('\n=== cuda_status executes ===');
  const statusTool = registered.get('cuda_status');
  const statusValue = await statusTool.execute({}, { signal: undefined, agent: undefined });
  const statusText = statusTool.output.render({}, statusValue)[0].text;
  console.log(statusText.split('\n').slice(0, 6).map((line) => '  ' + line).join('\n'));
  check('status finds the GPU', statusText.includes('RTX 4090'));
  check('status lists CUDA interpreters', statusText.includes('CUDA ok'));
  check('status points at the runs dir', statusText.includes('Recent jobs'));
  check('status returned structured jobs', Array.isArray(statusValue.jobs));

  /* ---------------------------------------------------------------- */
  console.log('\n=== cuda_run executes on the GPU ===');
  const runTool = registered.get('cuda_run');
  const runValue = await runTool.execute(
    {
      name: 'harness',
      timeoutSec: 120,
      code: [
        'import torch',
        'print("HARNESS_GPU_OK", torch.cuda.get_device_name(0))',
        'a = torch.randn(4096, 4096, device="cuda")',
        'b = torch.randn(4096, 4096, device="cuda")',
        'print("sum", float((a @ b).sum()))',
      ].join('\n'),
    },
    { signal: undefined, agent: { session: { header: { cwd: runsDir } } } },
  );
  const runText = runTool.output.render({}, runValue)[0].text;
  console.log(runText.split('\n').slice(0, 6).map((line) => '  ' + line).join('\n'));
  check('run finished', runValue.status === 'done', String(runValue.status));
  check('run printed the marker', (runValue.stdout ?? '').includes('HARNESS_GPU_OK'));
  check('run recorded peak VRAM', Number(runValue.facts?.peak_alloc_MB) > 0, `${runValue.facts?.peak_alloc_MB} MB`);
  check('rendered output names the device', runText.includes('RTX 4090') || runText.includes('peak VRAM'));

  /* ---------------------------------------------------------------- */
  console.log('\n=== cuda_job follows that run ===');
  const jobTool = registered.get('cuda_job');
  const jobValue = await jobTool.execute({ job: runValue.job, action: 'info' }, { signal: undefined });
  check('job info resolves by exact id', jobValue.status === 'done', String(jobValue.status));
  const prefix = await jobTool.execute({ job: runValue.job.slice(0, 15), action: 'logs' }, { signal: undefined });
  check('job logs resolve by prefix', (prefix.stdout ?? '').includes('HARNESS_GPU_OK'));
  const latest = await jobTool.execute({ job: 'latest', action: 'info' }, { signal: undefined });
  check('`latest` resolves', latest.job === runValue.job, String(latest.job));
  const badJob = await jobTool.execute({ job: 'nope-1999', action: 'info' }, { signal: undefined }).then(
    () => 'resolved',
    (error) => String(error.message),
  );
  check('unknown job fails with a clear message', badJob.includes('unknown job'), badJob.slice(0, 60));

  /* ---------------------------------------------------------------- */
  console.log('\n=== cuda_status lists the run ===');
  const after = await statusTool.execute({ jobs: 3 }, { signal: undefined });
  const afterText = statusTool.output.render({}, after)[0].text;
  check('the run shows up in the job list', afterText.includes(runValue.job));

  /* ---------------------------------------------------------------- */
  console.log('\n=== cuda_run passes arguments through ===');
  const argsProbe = await runTool.execute(
    {
      name: 'args-probe',
      timeoutSec: 60,
      args: ['a', 'b c'],
      code: 'import sys\nprint("ARGV", sys.argv[1:])',
    },
    { signal: undefined, agent: { session: { header: { cwd: runsDir } } } },
  );
  check('args reach sys.argv[1:]', (argsProbe.stdout ?? '').includes("['a', 'b c']"), (argsProbe.stdout ?? '').trim());
} finally {
  await rm(runsDir, { recursive: true, force: true }).catch(() => {});
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log('failed checks: ' + failures.join('; '));
  process.exit(1);
}
