/**
 * Proves the frontend toolchain can actually run before anything is started.
 *
 * Vite does not start itself: it hands the work to two native programs that npm
 * unpacks into node_modules - esbuild (esbuild.exe) and Rollup's .node addon.
 * When either one is damaged, Windows refuses to launch it with the
 * "side-by-side configuration is incorrect" dialog, the port stays dead, and
 * the reason is nowhere in the terminal output.
 *
 * So each native file is executed here for real. Exit code 0 means the dev
 * server can start; anything else prints one line saying what broke and what
 * to do about it.
 */
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(path.join(frontend, "package.json"));

/** STATUS_SIDE_BY_SIDE: the runtime is missing or the binary is damaged */
const SIDE_BY_SIDE = 0xc0000152;

/** One line, because start.bat shows this to somebody who just wants a demo. */
function fail(message, fix) {
  console.error(`FRONTEND CHECK FAILED: ${message}`);
  console.error(`FIX: ${fix}`);
  process.exit(1);
}

const reinstall = "delete frontend\\node_modules and run: npm --prefix frontend install";
const runtime = "install the Visual C++ 2015-2022 x64 runtime: winget install Microsoft.VisualStudio.2015-2022.Redist.14";

function esbuildBinary() {
  // the platform package is optional, so it has to be looked up rather than imported
  const pkg = `@esbuild/${process.platform}-${process.arch}`;
  try {
    return require.resolve(`${pkg}/package.json`).replace(/package\.json$/, "esbuild.exe");
  } catch {
    return null;
  }
}

function checkEsbuild() {
  const binary = esbuildBinary();
  if (!binary) {
    fail(`the esbuild package for ${process.platform}-${process.arch} is not installed`, reinstall);
  }
  if (!existsSync(binary)) {
    fail(`esbuild.exe is missing from ${path.relative(frontend, binary)}`, reinstall);
  }

  const run = spawnSync(binary, ["--version"], { encoding: "utf8" });
  if (run.error || run.status !== 0) {
    const sideBySide = run.status === SIDE_BY_SIDE || run.status === -1073741516;
    const detail = (run.stderr || run.error?.message || `exit code ${run.status}`).trim();
    if (sideBySide || /side-by-side/i.test(detail)) {
      fail(
        `esbuild.exe cannot run on this machine (${detail || "side-by-side configuration"})`,
        `${runtime}, then ${reinstall}`
      );
    }
    fail(`esbuild.exe did not run (${detail || `exit code ${run.status}`})`, reinstall);
  }
  console.log(`  esbuild  ${run.stdout.trim()}`);
}

function checkRollup() {
  // every native addon Rollup ships; at least one has to load
  const candidates = [
    `@rollup/rollup-win32-x64-msvc`,
    `@rollup/rollup-win32-x64-gnu`,
    `@rollup/rollup-darwin-arm64`,
    `@rollup/rollup-linux-x64-gnu`,
  ];
  let failures = [];
  for (const name of candidates) {
    try {
      const entry = require.resolve(name);
      require(entry);
      console.log(`  rollup   ${name} loaded`);
      return;
    } catch (err) {
      failures.push(`${name}: ${err.code || err.message}`);
    }
  }
  fail(`no Rollup native addon could be loaded (${failures.join("; ")})`, reinstall);
}

function checkNode() {
  const [major] = process.versions.node.split(".").map(Number);
  if (!major || major < 18) {
    fail(`Node.js ${process.versions.node} is too old, Vite 5 needs 18 or newer`, "install Node.js 20 LTS from https://nodejs.org");
  }
  console.log(`  node     v${process.versions.node}`);
}

console.log("Checking the frontend toolchain...");
checkNode();
checkEsbuild();
checkRollup();
console.log("Frontend toolchain is OK.");