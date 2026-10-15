#!/usr/bin/env node
/**
 * dsh-cuda lossless-JSON conformance test.
 *
 * DSH rejected the first live calls with `value is not lossless JSON` — a tool
 * value with `undefined` (or NaN, -0, a class instance, an array hole) cannot
 * cross the wire. The rule is `visitJsonValue` in `@deepseek-ai/dsh-api-gateway`;
 * this file carries a faithful copy of it, drives the real tools through a mock
 * context, and checks every value they return. Run it instead of restarting:
 *
 *   & "$env:USERPROFILE\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe" test\lossless.mjs
 */
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { apply } from '../lib/entry.js';

/* ------------------------------------------------------------------ *
 * The harness rule, copied from @deepseek-ai/dsh-api-gateway/lib/client.js
 * ------------------------------------------------------------------ */

function visitJsonValue(value, ancestors) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value) && !Object.is(value, -0);
  if (typeof value !== 'object') return false;
  if (ancestors.has(value)) return false;
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      if (Object.getPrototypeOf(value) !== Array.prototype) return false;
      if (Reflect.ownKeys(value).length !== value.length + 1) return false;
      for (let index = 0; index < value.length; index++) {
        if (!Object.hasOwn(value, index) || !visitJsonValue(value[index], ancestors)) return false;
      }
      return true;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return false;
    for (const key of Reflect.ownKeys(value)) {
      if (typeof key !== 'string') return false;
      if (Object.getOwnPropertyDescriptor(value, key)?.enumerable !== true) return false;
      if (!visitJsonValue(Reflect.get(value, key), ancestors)) return false;
    }
    return true;
  } finally {
    ancestors.delete(value);
  }
}

function isLosslessJson(value) {
  return visitJsonValue(value, new Set());
}

/** Walk a value the way the rule does and name the first offending path. */
function firstOffence(value, path = '$', ancestors = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return null;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return `${path} = ${value}`;
    if (Object.is(value, -0)) return `${path} = -0`;
    return null;
  }
  if (value === undefined) return `${path} is undefined`;
  if (typeof value !== 'object') return `${path} is a ${typeof value}`;
  if (ancestors.has(value)) return `${path} is a cycle`;
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      if (Object.getPrototypeOf(value) !== Array.prototype) return `${path} is not a plain array`;
      for (let index = 0; index < value.length; index++) {
        if (!Object.hasOwn(value, index)) return `${path}[${index}] is a hole`;
        const bad = firstOffence(value[index], `${path}[${index}]`, ancestors);
        if (bad) return bad;
      }
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      return `${path} is a ${value.constructor?.name ?? 'class instance'}`;
    }
    for (const key of Object.keys(value)) {
      const bad = firstOffence(value[key], `${path}.${key}`, ancestors);
      if (bad) return bad;
    }
    return null;
  } finally {
    ancestors.delete(value);
  }
}

/* ------------------------------------------------------------------ *
 * Mock context
 * ------------------------------------------------------------------ */

const registered = new Map();
const skillsCtx = { effect: (fn) => fn(), skills: { registerProvider: () => () => {} } };
const ctx = {
  effect: (fn) => fn(),
  tools: {
    register(definition) {
      registered.set(definition.name, definition);
      return () => registered.delete(definition.name);
    },
  },
  inject: (_deps, callback) => callback(skillsCtx),
  logger: { warn: () => {} },
};

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

/** Run one tool call and assert both that it worked and that its value travels. */
async function probe(label, toolName, args, exec = { signal: undefined }) {
  const definition = registered.get(toolName);
  if (!definition) {
    check(label, false, `${toolName} is not registered`);
    return undefined;
  }
  let value;
  try {
    value = await definition.execute(args, exec);
  } catch (error) {
    check(`${label} — ${toolName} runs`, false, String(error?.message ?? error).slice(0, 120));
    return undefined;
  }
  const offence = firstOffence(value);
  check(`${label} — ${toolName} returns lossless JSON`, offence === null, offence ?? 'clean');
  // The renderer must survive the scrubbed value too.
  try {
    const blocks = definition.output.render(args, value);
    check(`${label} — ${toolName} renders`, Array.isArray(blocks) && typeof blocks[0]?.text === 'string');
  } catch (error) {
    check(`${label} — ${toolName} renders`, false, String(error?.message ?? error).slice(0, 120));
  }
  return value;
}

const runsDir = await mkdtemp(join(tmpdir(), 'dsh-cuda-lossless-'));
console.log(`runs dir : ${runsDir}`);

try {
  apply(ctx, { runsDir });
  check('three tools registered', registered.size === 3, [...registered.keys()].join(', '));

  console.log('\n=== every branch must return lossless JSON ===');
  await probe('status', 'cuda_status', {}, { signal: undefined });
  await probe('status with jobs', 'cuda_status', { jobs: 3, refresh: false }, { signal: undefined });

  const run = await probe('foreground run', 'cuda_run', {
    name: 'lossless',
    timeoutSec: 120,
    code: 'import torch\nprint("LOSSLESS_OK", float(torch.ones(4, device="cuda").sum()))',
  }, { signal: undefined, agent: { session: { header: { cwd: runsDir } } } });

  await probe('background run', 'cuda_run', {
    name: 'lossless-bg',
    background: true,
    code: 'import time\nprint("bg", flush=True)\ntime.sleep(2)\n',
  }, { signal: undefined, agent: { session: { header: { cwd: runsDir } } } });

  if (run?.job) {
    await probe('job info', 'cuda_job', { job: run.job, action: 'info' });
    await probe('job logs', 'cuda_job', { job: run.job, action: 'logs', stream: 'both' });
    await probe('job wait', 'cuda_job', { job: run.job, action: 'wait', waitMs: 5_000 });
    await probe('job by prefix', 'cuda_job', { job: run.job.slice(0, 15), action: 'info' });
    await probe('job latest', 'cuda_job', { job: 'latest', action: 'info' });
  }

  console.log('\n=== the rule itself has teeth ===');
  const planted = [
    ['undefined', { a: undefined }],
    ['NaN', { a: Number.NaN }],
    ['negative zero', { a: -0 }],
    ['a class instance', { a: new Date() }],
    ['an array hole', [1, , 3]],
    ['a cycle', (() => { const c = {}; c.self = c; return c; })()],
  ];
  for (const [label, value] of planted) {
    check(`rule rejects ${label}`, firstOffence(value) !== null && isLosslessJson(value) === false, firstOffence(value) ?? 'accepted!');
  }
  check('rule accepts a clean value', isLosslessJson({ a: [1, 'x', true, null] }) === true);

  console.log('\n=== the scrubber itself ===');
  const statusValue = await registered.get('cuda_status').execute({}, { signal: undefined });
  check('no undefined survives in the status value', firstOffence(statusValue) === null, firstOffence(statusValue) ?? 'clean');
  check('gpuError is absent rather than undefined', !('gpuError' in statusValue) || statusValue.gpuError !== undefined);
} finally {
  await rm(runsDir, { recursive: true, force: true }).catch(() => {});
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log('failed checks: ' + failures.join('; '));
  process.exit(1);
}
