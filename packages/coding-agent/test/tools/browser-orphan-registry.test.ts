/**
 * Regression tests for issue #10022: the project-shared broker-owned Chromium
 * (`omp.browser.headless`) retains page targets created by omp processes that
 * ended abnormally, because tab ownership was tracked only in per-process
 * memory. `orphan-registry` records ownership durably and reaps targets whose
 * owning process is gone.
 *
 * The contract under test:
 *  - a dead owner's targets are collected for reaping, a live owner's are not;
 *  - this process's own targets are never reaped (a live session keeps its tabs);
 *  - a conservative grace window keeps a just-crashed owner's fresh records;
 *  - a close counts only once Chromium's own `Target.getTargets` stops listing
 *    the target — not on `success: true`, not on puppeteer's view — and is
 *    re-sent until then;
 *  - confirmed closures are removed, while unconfirmed ones remain durable and
 *    are retried (by the next reap, or for a live owner by its unclosed retry).
 */
import { afterEach, describe, expect, it, spyOn, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as piUtils from "@oh-my-pi/pi-utils";
import { daemonRuntimeDir } from "@oh-my-pi/pi-coding-agent/launch/paths";
import * as launch from "@oh-my-pi/pi-coding-agent/tools/browser/launch";
import * as orphanRegistry from "@oh-my-pi/pi-coding-agent/tools/browser/orphan-registry";
import {
	closeSharedTargetUntilGone,
	collectOrphanTargets,
	forgetSharedTarget,
	markSharedTargetUnclosed,
	reapOrphanSharedTargets,
	recordSharedTarget,
	resetOrphanRegistryForTest,
	retryUnclosedSharedTargets,
	type SharedTargetScope,
} from "@oh-my-pi/pi-coding-agent/tools/browser/orphan-registry";
import * as registry from "@oh-my-pi/pi-coding-agent/tools/browser/registry";
import * as sharedDaemon from "@oh-my-pi/pi-coding-agent/tools/browser/shared-daemon";
import type { Browser, default as Puppeteer } from "puppeteer-core";

const DAEMON_NAME = "omp.browser.headless";

/** Unique per-test scope so registry dirs never collide across the suite. */
function makeScope(): SharedTargetScope {
	const projectDir = path.join("/tmp", `omp-orphan-test-${crypto.randomUUID()}`);
	return { projectDir, daemonName: DAEMON_NAME };
}

function registryDir(scope: SharedTargetScope): string {
	return path.join(daemonRuntimeDir(scope.projectDir), `${scope.daemonName}.targets`);
}

async function writeOwnershipFile(
	scope: SharedTargetScope,
	record: { pid: number; updatedAt: number; targets: string[] },
): Promise<void> {
	const dir = registryDir(scope);
	await fs.mkdir(dir, { recursive: true });
	await Bun.write(path.join(dir, `${record.pid}.json`), JSON.stringify(record));
}

/** A pid that has been spawned and reaped, so `kill(pid, 0)` reports ESRCH. */
async function deadPid(): Promise<number> {
	// Bun's own executable is the only binary guaranteed on every platform (`true` is absent on Windows).
	const proc = Bun.spawn([process.execPath, "-e", ""], { stdout: "ignore", stderr: "ignore" });
	await proc.exited;
	return proc.pid;
}

/** Fake Chromium behind a puppeteer `Browser` stub; `targets` is Chromium's own view. */
interface FakeChromium {
	browser: Browser;
	/** Target ids Chromium still lists in `Target.getTargets`. */
	targets: Set<string>;
	/** `Target.closeTarget` sends per target id. */
	closeSends: Map<string, number>;
	/** Close sends answered `success: true` while the target stays listed, per id (`Infinity`: never goes). */
	linger: Map<string, number>;
	/** `Target.getTargets` sends. */
	listings: number;
	/** `createCDPSession()` calls on the browser target; closes must never attach. */
	sessions: number;
	/** Drop the connection: puppeteer then reports `connected: false` and rejects every send. */
	disconnect(): void;
}

/**
 * Chromium answers every close of a listed target with `success: true`; a
 * lingering target survives its first `linger` sends, as seen on pages aborted
 * mid-navigation. Ids in `failClose` reject the close and stay listed (a
 * transient CDP failure); ids in `rejectClose` reject every close, yet past
 * their `linger` sends the target goes anyway (the reply was lost); the first
 * `hangClose` sends of an id never answer. Puppeteer's own view (`targets()`)
 * is always empty, as after its session detached, so confirming through it
 * would wrongly report every target gone. Browser-level commands arrive on the
 * root connection (`_connection`, what `browserConnection` reads); attaching
 * to the browser target fails.
 */
function makeChromium(
	opts: {
		targets?: string[];
		linger?: Record<string, number>;
		failClose?: ReadonlySet<string>;
		rejectClose?: ReadonlySet<string>;
		hangClose?: Record<string, number>;
		onCloseSend?: (targetId: string) => void;
	} = {},
): FakeChromium {
	let connected = true;
	const chromium: FakeChromium = {
		browser: undefined as unknown as Browser,
		targets: new Set(opts.targets ?? []),
		closeSends: new Map(),
		linger: new Map(Object.entries(opts.linger ?? {})),
		listings: 0,
		sessions: 0,
		disconnect: () => {
			connected = false;
		},
	};
	const connection = {
		send: async (method: string, params?: { targetId: string }) => {
			if (!connected) throw new Error("Protocol error: Connection closed.");
			if (method === "Target.getTargets") {
				chromium.listings++;
				return { targetInfos: [...chromium.targets].map(targetId => ({ targetId, type: "page" })) };
			}
			if (method !== "Target.closeTarget" || !params) throw new Error(`Unexpected ${method}`);
			const { targetId } = params;
			const sends = (chromium.closeSends.get(targetId) ?? 0) + 1;
			chromium.closeSends.set(targetId, sends);
			opts.onCloseSend?.(targetId);
			if (sends <= (opts.hangClose?.[targetId] ?? 0)) return await new Promise<never>(() => {});
			if (opts.failClose?.has(targetId)) throw new Error("transient CDP failure");
			if (!chromium.targets.has(targetId)) throw new Error("No target with given id found");
			if (sends > (chromium.linger.get(targetId) ?? 0)) chromium.targets.delete(targetId);
			if (opts.rejectClose?.has(targetId)) throw new Error("Session closed");
			return { success: true };
		},
		createSession: async () => {
			throw new Error("closes must not attach to targets");
		},
	};
	chromium.browser = {
		get connected() {
			return connected;
		},
		_connection: connection,
		target: () => ({
			createCDPSession: async () => {
				chromium.sessions++;
				throw new Error("closes must not attach to the browser target");
			},
		}),
		targets: () => [],
		on: () => undefined,
		off: () => undefined,
		disconnect: () => chromium.disconnect(),
	} as unknown as Browser;
	return chromium;
}

/** Fast close bounds for tests; the production defaults are seconds. */
const FAST = { intervalMs: 20, budgetMs: 2_000 };
/** Budget for targets that never go, so a give-up is quick. */
const GIVE_UP = { intervalMs: 20, budgetMs: 150 };

async function ownedTargets(scope: SharedTargetScope): Promise<string[] | undefined> {
	const file = Bun.file(path.join(registryDir(scope), `${process.pid}.json`));
	if (!(await file.exists())) return undefined;
	return ((await file.json()) as { targets: string[] }).targets;
}

const scopes: SharedTargetScope[] = [];
function trackedScope(): SharedTargetScope {
	const scope = makeScope();
	scopes.push(scope);
	return scope;
}

afterEach(async () => {
	vi.restoreAllMocks();
	resetOrphanRegistryForTest();
	for (const scope of scopes.splice(0)) {
		await fs.rm(daemonRuntimeDir(scope.projectDir), { recursive: true, force: true }).catch(() => undefined);
	}
});

describe("orphan-registry — ownership scan", () => {
	it("collects a dead owner's targets and leaves a live owner's untouched", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		const live = 424242; // treated as alive by the injected probe below
		await writeOwnershipFile(scope, { pid: dead, updatedAt: 0, targets: ["dead-a", "dead-b"] });
		await writeOwnershipFile(scope, { pid: live, updatedAt: 0, targets: ["live-a"] });

		const scan = await collectOrphanTargets(scope, {
			now: () => 10_000_000,
			isAlive: pid => pid === live,
		});

		expect(scan.owners).toEqual([
			{
				file: path.join(registryDir(scope), `${dead}.json`),
				pid: dead,
				updatedAt: 0,
				targetIds: ["dead-a", "dead-b"],
			},
		]);
	});

	it("never reaps this process's own recorded targets", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "mine-1");
		await recordSharedTarget(scope, "mine-2");

		// Our own file is present on disk...
		const own = (await Bun.file(path.join(registryDir(scope), `${process.pid}.json`)).json()) as {
			targets: string[];
		};
		expect(own.targets.sort()).toEqual(["mine-1", "mine-2"]);

		// ...but a scan (even with everything else forced dead) skips it.
		const scan = await collectOrphanTargets(scope, { now: () => 10_000_000, isAlive: () => false });
		expect(scan.owners).toEqual([]);
	});

	it("forgetSharedTarget drops one id and removes the file once empty", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "a");
		await recordSharedTarget(scope, "b");
		await forgetSharedTarget(scope, "a");
		const after = (await Bun.file(path.join(registryDir(scope), `${process.pid}.json`)).json()) as {
			targets: string[];
		};
		expect(after.targets).toEqual(["b"]);

		await forgetSharedTarget(scope, "b");
		expect(await Bun.file(path.join(registryDir(scope), `${process.pid}.json`)).exists()).toBe(false);
	});

	it("keeps a dead owner's records inside the conservative grace window", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		await writeOwnershipFile(scope, { pid: dead, updatedAt: 100_000, targets: ["fresh"] });

		const withinGrace = await collectOrphanTargets(scope, {
			now: () => 105_000,
			isAlive: () => false,
			graceMs: 15_000,
		});
		expect(withinGrace.owners).toEqual([]);

		const pastGrace = await collectOrphanTargets(scope, {
			now: () => 130_000,
			isAlive: () => false,
			graceMs: 15_000,
		});
		expect(pastGrace.owners).toEqual([
			{
				file: path.join(registryDir(scope), `${dead}.json`),
				pid: dead,
				updatedAt: 100_000,
				targetIds: ["fresh"],
			},
		]);
	});
});

describe("orphan-registry — closeSharedTargetUntilGone", () => {
	it("re-sends the close until Chromium stops listing a target that answered success", async () => {
		const chromium = makeChromium({ targets: ["lingering"], linger: { lingering: 2 } });

		expect(await closeSharedTargetUntilGone(chromium.browser, "lingering", FAST)).toBe(true);

		// Two `success: true` replies left it listed; only the third send removed it.
		expect(chromium.closeSends.get("lingering")).toBe(3);
		expect(chromium.targets.has("lingering")).toBe(false);
		expect(chromium.sessions).toBe(0);
	});

	it("returns false within its budget while Chromium keeps listing the target", async () => {
		const chromium = makeChromium({ targets: ["stuck"], linger: { stuck: Infinity } });
		const started = performance.now();

		expect(await closeSharedTargetUntilGone(chromium.browser, "stuck", GIVE_UP)).toBe(false);

		expect(performance.now() - started).toBeLessThan(1_000);
		expect(chromium.closeSends.get("stuck")).toBeGreaterThan(1);
		expect(chromium.targets.has("stuck")).toBe(true);
	});

	it("confirms a target whose close rejected once Chromium no longer lists it", async () => {
		const chromium = makeChromium({
			targets: ["rejecting"],
			linger: { rejecting: 1 },
			rejectClose: new Set(["rejecting"]),
		});

		expect(await closeSharedTargetUntilGone(chromium.browser, "rejecting", FAST)).toBe(true);

		// The first rejected send left it listed, so the close was re-sent.
		expect(chromium.closeSends.get("rejecting")).toBe(2);
		expect(chromium.targets.has("rejecting")).toBe(false);
	});

	it("returns false when its signal aborts before the target goes", async () => {
		const chromium = makeChromium({ targets: ["stuck"], linger: { stuck: Infinity } });
		const controller = new AbortController();
		const closing = closeSharedTargetUntilGone(chromium.browser, "stuck", {
			intervalMs: 20,
			budgetMs: 60_000,
			signal: controller.signal,
		});
		controller.abort();

		expect(await closing).toBe(false);
		expect(chromium.targets.has("stuck")).toBe(true);
	});

	it("runs concurrent closes over the root connection without attaching a session", async () => {
		// Concurrent browser-target sessions made puppeteer 25.3.0 drop the browser target,
		// so every later web search failed with `Browser target is not found`.
		const ids = Array.from({ length: 10 }, (_, i) => `page-${i}`);
		const chromium = makeChromium({ targets: ids, linger: { "page-3": 1, "page-7": 2 } });

		const results = await Promise.all(ids.map(id => closeSharedTargetUntilGone(chromium.browser, id, FAST)));

		expect(results).toEqual(ids.map(() => true));
		expect(chromium.targets.size).toBe(0);
		expect(chromium.sessions).toBe(0);
	});

	it("re-sends a close that never answers at the next interval", async () => {
		const chromium = makeChromium({ targets: ["hung"], hangClose: { hung: 1 } });

		expect(await closeSharedTargetUntilGone(chromium.browser, "hung", FAST)).toBe(true);

		expect(chromium.closeSends.get("hung")).toBe(2);
		expect(chromium.targets.has("hung")).toBe(false);
	});

	it("stops as soon as the connection drops instead of spinning to its budget", async () => {
		const chromium: FakeChromium = makeChromium({
			targets: ["stuck"],
			linger: { stuck: Infinity },
			onCloseSend: () => chromium.disconnect(),
		});
		const started = performance.now();

		expect(await closeSharedTargetUntilGone(chromium.browser, "stuck", { intervalMs: 20, budgetMs: 60_000 })).toBe(
			false,
		);

		expect(performance.now() - started).toBeLessThan(1_000);
		// One listing and one close; nothing is sent over the dropped connection.
		expect(chromium.listings).toBe(1);
		expect(chromium.closeSends.get("stuck")).toBe(1);
	});
});

describe("orphan-registry — unclosed retry", () => {
	it("closes a marked lingering target, forgets its record, and unmarks it", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "lingering");
		await recordSharedTarget(scope, "open-tab");
		markSharedTargetUnclosed(scope, "lingering");
		const chromium = makeChromium({ targets: ["lingering", "open-tab"], linger: { lingering: 1 } });

		expect(await retryUnclosedSharedTargets(chromium.browser, scope, FAST)).toBe(1);

		expect(chromium.closeSends.get("lingering")).toBe(2);
		expect(chromium.targets.has("lingering")).toBe(false);
		expect(await ownedTargets(scope)).toEqual(["open-tab"]);
		// Unmarked: a second retry sends nothing, and an unmarked live tab is never touched.
		expect(await retryUnclosedSharedTargets(chromium.browser, scope, FAST)).toBe(0);
		expect(chromium.closeSends.get("lingering")).toBe(2);
		expect(chromium.closeSends.has("open-tab")).toBe(false);
	});

	it("keeps the record and the mark while the target still lingers", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "stuck");
		markSharedTargetUnclosed(scope, "stuck");
		const chromium = makeChromium({ targets: ["stuck"], linger: { stuck: Infinity } });

		expect(await retryUnclosedSharedTargets(chromium.browser, scope, GIVE_UP)).toBe(0);
		expect(chromium.targets.has("stuck")).toBe(true);
		expect(await ownedTargets(scope)).toEqual(["stuck"]);

		// Still marked: once Chromium lets it go, the next retry closes and forgets it.
		chromium.linger.delete("stuck");
		expect(await retryUnclosedSharedTargets(chromium.browser, scope, FAST)).toBe(1);
		expect(await ownedTargets(scope)).toBeUndefined();
	});

	it("coalesces concurrent retries for one scope into a single run", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "lingering");
		markSharedTargetUnclosed(scope, "lingering");
		const chromium = makeChromium({ targets: ["lingering"], linger: { lingering: 1 } });

		const first = retryUnclosedSharedTargets(chromium.browser, scope, FAST);
		const joined = retryUnclosedSharedTargets(chromium.browser, scope, FAST);

		expect(joined).toBe(first);
		expect(await Promise.all([first, joined])).toEqual([1, 1]);
		expect(chromium.closeSends.get("lingering")).toBe(2);
	});
});

describe("orphan-registry — reap", () => {
	it("closes a dead owner's targets via CDP and deletes its ownership file", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		await writeOwnershipFile(scope, {
			pid: dead,
			updatedAt: Date.now() - 60_000,
			targets: ["orphan-1", "orphan-2"],
		});
		const chromium = makeChromium({ targets: ["orphan-1", "orphan-2"] });

		const count = await reapOrphanSharedTargets(chromium.browser, scope, FAST);

		expect(count).toBe(2);
		expect([...chromium.closeSends.keys()].sort()).toEqual(["orphan-1", "orphan-2"]);
		expect(chromium.targets.size).toBe(0);
		expect(await Bun.file(path.join(registryDir(scope), `${dead}.json`)).exists()).toBe(false);
	});

	it("retains transient CDP failures and retries them on the next reap", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		const updatedAt = Date.now() - 60_000;
		await writeOwnershipFile(scope, {
			pid: dead,
			updatedAt,
			targets: ["closed-now", "retry-later"],
		});
		const first = makeChromium({ targets: ["closed-now", "retry-later"], failClose: new Set(["retry-later"]) });

		const firstCount = await reapOrphanSharedTargets(first.browser, scope, GIVE_UP);

		expect(firstCount).toBe(1);
		expect(first.targets.has("closed-now")).toBe(false);
		const retained = (await Bun.file(path.join(registryDir(scope), `${dead}.json`)).json()) as {
			pid: number;
			updatedAt: number;
			targets: string[];
		};
		expect(retained).toEqual({ pid: dead, updatedAt, targets: ["retry-later"] });

		const retry = makeChromium({ targets: ["retry-later"] });
		const retryCount = await reapOrphanSharedTargets(retry.browser, scope, FAST);

		expect(retryCount).toBe(1);
		expect([...retry.closeSends.keys()]).toEqual(["retry-later"]);
		expect(await Bun.file(path.join(registryDir(scope), `${dead}.json`)).exists()).toBe(false);
	});

	it("keeps a lingering orphan's record when its close answered success but Chromium still lists it", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		const updatedAt = Date.now() - 60_000;
		await writeOwnershipFile(scope, { pid: dead, updatedAt, targets: ["gone-now", "lingering"] });
		const chromium = makeChromium({ targets: ["gone-now", "lingering"], linger: { lingering: Infinity } });

		expect(await reapOrphanSharedTargets(chromium.browser, scope, GIVE_UP)).toBe(1);

		expect(chromium.targets.has("lingering")).toBe(true);
		const retained = (await Bun.file(path.join(registryDir(scope), `${dead}.json`)).json()) as {
			targets: string[];
		};
		expect(retained.targets).toEqual(["lingering"]);
	});
});

describe("orphan-registry — shared-browser attach", () => {
	it("retries this process's unclosed targets when it attaches to the shared browser", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "lingering");
		await recordSharedTarget(scope, "open-tab");
		markSharedTargetUnclosed(scope, "lingering");
		const chromium = makeChromium({ targets: ["lingering", "open-tab"], linger: { lingering: 1 } });
		// A CLI worker host attaches to the broker-owned Chromium (`openSharedHeadlessHandle`).
		spyOn(piUtils, "workerHostEntry").mockReturnValue("/omp/cli.ts");
		spyOn(sharedDaemon, "ensureSharedBrowser").mockResolvedValue({
			wsEndpoint: "ws://127.0.0.1:9/devtools/browser/fake",
			daemonName: scope.daemonName,
			projectDir: scope.projectDir,
		});
		spyOn(launch, "loadPuppeteer").mockResolvedValue({} as typeof Puppeteer);
		spyOn(launch, "connectPuppeteer").mockResolvedValue(chromium.browser);
		const retry = spyOn(orphanRegistry, "retryUnclosedSharedTargets");

		const handle = await registry.acquireBrowser({ kind: "headless", headless: true }, { cwd: scope.projectDir });
		try {
			// The attach starts the retry detached; await the run it started.
			expect(retry).toHaveBeenCalledTimes(1);
			expect(retry.mock.calls[0]?.[1]).toEqual(scope);
			expect(await retry.mock.results[0]?.value).toBe(1);

			expect(chromium.targets.has("lingering")).toBe(false);
			expect(chromium.closeSends.has("open-tab")).toBe(false);
			expect(await ownedTargets(scope)).toEqual(["open-tab"]);
		} finally {
			registry.holdBrowser(handle);
			await registry.releaseBrowser(handle, { kill: false });
		}
	});
});
