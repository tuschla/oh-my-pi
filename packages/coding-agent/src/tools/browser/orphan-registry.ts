/**
 * Durable ownership registry for targets in the project-shared broker-owned
 * Chromium (`omp.browser.headless`/`omp.browser.headed`).
 *
 * The shared browser outlives any single omp process, but tab ownership is
 * otherwise tracked only in that process's memory (`tab-supervisor`'s `tabs`
 * map). When a session ends abnormally (crash, SIGKILL, cleanup timeout) its
 * in-process map dies with it and the pages it opened stay open in the shared
 * Chromium forever, accumulating into multi-GB orphan targets (issue #10022).
 *
 * This module records, on disk under the broker runtime dir, which OS process
 * created each shared-browser page target. Any live omp process can then reap
 * targets whose owning process is gone. It only ever touches OMP-owned
 * shared-browser targets — user-owned connected/relay/spawned browsers have no
 * registry and are never scanned.
 *
 * Ownership is authoritative in the safe direction: a target is reaped only
 * when its owner PID reports `ESRCH` (definitively dead). A live PID is never
 * reaped, so a live session's tabs cannot be yanked out from under it; the
 * worst case (recycled PID) leaves an orphan uncollected rather than closing a
 * live page.
 */
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { isEnoent, logger, untilAborted } from "@oh-my-pi/pi-utils";
import type { Browser, CDPSession, Connection } from "puppeteer-core";
import { daemonRuntimeDir } from "../../launch/paths";
import { browserConnection } from "./launch";

/** Identifies one shared-browser daemon's target registry. */
export interface SharedTargetScope {
	/** Canonical project directory owning the broker (as stamped on the handle). */
	projectDir: string;
	/** Broker daemon name, e.g. `omp.browser.headless`. */
	daemonName: string;
}

/** On-disk ownership record: one file per owning omp process. */
interface OwnershipFile {
	pid: number;
	updatedAt: number;
	targets: string[];
}

/**
 * Reap only records whose owner has been dead AND untouched for this long.
 * The PID probe is already authoritative; the grace window is a conservative
 * guard against clock skew and PID reuse races, and keeps a just-crashed
 * process's very fresh records around briefly in case it is being restarted.
 */
const DEFAULT_GRACE_MS = 15_000;

/**
 * Re-send cadence and overall budget of {@link closeSharedTargetUntilGone}.
 * Chromium can answer `Target.closeTarget` with `success: true` and keep the
 * page (seen on pages aborted mid-navigation; they survived a further minute),
 * while a re-sent close removes them within ~1.5 s.
 */
const RECLOSE_INTERVAL_MS = 2_000;
const RECLOSE_BUDGET_MS = 30_000;
/**
 * Bounds of the backoff between re-reads of Chromium's target list after a
 * close: Chromium drops a closed target within milliseconds, so the first
 * re-read comes soon after each send, then they back off while it lingers.
 */
const CLOSE_POLL_MIN_MS = 10;
const CLOSE_POLL_MAX_MS = 250;
/**
 * Per-target budget of the detached sweeps (orphan reap, unclosed retry), so a
 * sweep over many lingering targets stays bounded; an unconfirmed target keeps
 * its record and is retried by the next sweep.
 */
const SWEEP_CLOSE_BUDGET_MS = 5_000;

/** In-process set of shared-browser targets this process created, keyed by registry dir. */
const ownedByDir = new Map<string, Set<string>>();
/** Owned targets whose close this process could not confirm, keyed by registry dir; retried on attach. */
const unclosedByDir = new Map<string, Set<string>>();
/** In-flight {@link retryUnclosedSharedTargets} runs per registry dir (single-flight). */
const unclosedRetries = new Map<string, Promise<number>>();
/** Per-registry-dir write serialization so concurrent record/forget can't tear the file. */
const writeChains = new Map<string, Promise<void>>();

function registryDir(scope: SharedTargetScope): string {
	return path.join(daemonRuntimeDir(scope.projectDir), `${scope.daemonName}.targets`);
}

/** Serialize a write against others for the same registry dir. */
function chain(dir: string, task: () => Promise<void>): Promise<void> {
	const prev = writeChains.get(dir) ?? Promise.resolve();
	const next = prev.then(task, task);
	writeChains.set(
		dir,
		next.catch(() => undefined),
	);
	return next;
}

/** Persist (or, when empty, remove) this process's ownership file for a registry dir. */
async function flush(dir: string): Promise<void> {
	const owned = ownedByDir.get(dir);
	const file = path.join(dir, `${process.pid}.json`);
	if (!owned || owned.size === 0) {
		await fs.rm(file, { force: true }).catch(() => undefined);
		return;
	}
	const record: OwnershipFile = { pid: process.pid, updatedAt: Date.now(), targets: [...owned] };
	const tmp = `${file}.${process.pid}.tmp`;
	await fs.mkdir(dir, { recursive: true });
	await Bun.write(tmp, JSON.stringify(record));
	await fs.rename(tmp, file);
}

/** Record that this process created `targetId` in the given shared browser. */
export async function recordSharedTarget(scope: SharedTargetScope, targetId: string): Promise<void> {
	const dir = registryDir(scope);
	let owned = ownedByDir.get(dir);
	if (!owned) {
		owned = new Set();
		ownedByDir.set(dir, owned);
	}
	owned.add(targetId);
	await chain(dir, () => flush(dir)).catch(err =>
		logger.debug("Failed to record shared-browser target ownership", {
			error: err instanceof Error ? err.message : String(err),
		}),
	);
}

/** Drop `targetId` from this process's ownership file (closed the normal way). */
export async function forgetSharedTarget(scope: SharedTargetScope, targetId: string): Promise<void> {
	const dir = registryDir(scope);
	unclosedByDir.get(dir)?.delete(targetId);
	const owned = ownedByDir.get(dir);
	if (!owned?.delete(targetId)) return;
	await chain(dir, () => flush(dir)).catch(err =>
		logger.debug("Failed to update shared-browser target ownership", {
			error: err instanceof Error ? err.message : String(err),
		}),
	);
}

/**
 * Drop every target this process still claims in `scope`. For when the whole
 * shared browser is confirmed gone — the broker reported its daemon in a
 * terminal state after a stop — which takes the targets that could not be
 * closed with it; the records are dead weight that would otherwise be
 * rewritten on every later write until this process exits. Only call this on
 * that confirmation: while the browser lives, the records are the only handle
 * a later reap has on its orphaned targets.
 */
export async function forgetSharedTargets(scope: SharedTargetScope): Promise<void> {
	const dir = registryDir(scope);
	unclosedByDir.delete(dir);
	const owned = ownedByDir.get(dir);
	if (!owned || owned.size === 0) return;
	owned.clear();
	await chain(dir, () => flush(dir)).catch(err =>
		logger.debug("Failed to clear shared-browser target ownership", {
			error: err instanceof Error ? err.message : String(err),
		}),
	);
}

/** True when `pid` names a live process; non-`ESRCH` probe failures are treated as alive (safe direction). */
function isPidAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return (error as NodeJS.ErrnoException).code !== "ESRCH";
	}
}

/** Options for {@link collectOrphanTargets}; the defaults hit the real registry, the seams are for tests. */
export interface CollectOrphanOptions {
	/** Wall clock; injectable for deterministic grace-window tests. */
	now?: () => number;
	/** PID liveness probe; injectable so tests need no real subprocesses. */
	isAlive?: (pid: number) => boolean;
	/** Grace window in ms before a dead owner's records are eligible. */
	graceMs?: number;
}

/** Targets belonging to one dead process, kept grouped so partial failures remain retryable. */
export interface OrphanOwner {
	file: string;
	pid: number;
	updatedAt: number;
	targetIds: string[];
}

/** Orphan-scan result grouped by durable ownership file. */
export interface OrphanScan {
	owners: OrphanOwner[];
}

/**
 * Scan a registry dir for targets whose owning process is gone. Returns one
 * entry per dead owner so a reaper can retain only targets whose CDP closure
 * was not confirmed. This process's own file and every live-owner file are
 * left untouched.
 */
export async function collectOrphanTargets(
	scope: SharedTargetScope,
	opts: CollectOrphanOptions = {},
): Promise<OrphanScan> {
	const now = opts.now ?? Date.now;
	const isAlive = opts.isAlive ?? isPidAlive;
	const graceMs = opts.graceMs ?? DEFAULT_GRACE_MS;
	const dir = registryDir(scope);
	let entries: string[];
	try {
		entries = await fs.readdir(dir);
	} catch (err) {
		if (isEnoent(err)) return { owners: [] };
		throw err;
	}
	const owners: OrphanOwner[] = [];
	const nowMs = now();
	for (const entry of entries) {
		if (!entry.endsWith(".json")) continue;
		const file = path.join(dir, entry);
		let record: OwnershipFile;
		try {
			record = (await Bun.file(file).json()) as OwnershipFile;
		} catch {
			continue; // torn or malformed file; a live owner will rewrite it
		}
		if (typeof record?.pid !== "number" || !Array.isArray(record.targets)) continue;
		if (record.pid === process.pid) continue; // our own file
		if (isAlive(record.pid)) continue; // owner still running
		if (nowMs - (record.updatedAt ?? 0) < graceMs) continue; // conservative grace
		owners.push({
			file,
			pid: record.pid,
			updatedAt: record.updatedAt,
			targetIds: record.targets.filter(id => typeof id === "string"),
		});
	}
	return { owners };
}

/**
 * Close a page target by id through a fresh CDP session. Returns true only
 * when CDP confirms the close or confirms the target no longer exists; a
 * dropped connection or transient protocol failure returns false so durable
 * ownership remains available for a later retry.
 */
export async function closeCdpTarget(browser: Browser, targetId: string): Promise<boolean> {
	const session = await browser
		.target()
		.createCDPSession()
		.catch(() => null);
	if (!session) return false;
	try {
		try {
			const result = await session.send("Target.closeTarget", { targetId });
			if (result.success) return true;
		} catch {
			// A concurrent reaper or the page itself may already have closed the
			// target. Confirm absence before treating the cleanup as complete.
		}
		try {
			const { targetInfos } = await session.send("Target.getTargets");
			return !targetInfos.some(info => info.targetId === targetId);
		} catch {
			return false;
		}
	} finally {
		await session.detach().catch(() => undefined);
	}
}

/**
 * {@link closeCdpTarget} for the omp relay, over a channel that answers browser-level `Target.*`
 * commands. The relay's `Target.getTargets` lists only tabs it can drive right now, not those of a
 * disconnected browser or with a dismissed debugger, so absence is read from `Target.getTargetInfo`:
 * it looks the id up among every tab the extensions reported, and answers an unknown id with the
 * browser target.
 */
export async function closeRelayTarget(channel: Pick<CDPSession, "send">, targetId: string): Promise<boolean> {
	try {
		const result = await channel.send("Target.closeTarget", { targetId });
		if (result.success) return true;
	} catch {
		// Closed already (the worker's own close), or the extension cannot answer.
	}
	try {
		const { targetInfo } = await channel.send("Target.getTargetInfo", { targetId });
		return targetInfo.targetId !== targetId;
	} catch {
		return false;
	}
}

/** Options for {@link closeSharedTargetUntilGone}. */
export interface CloseUntilGoneOptions {
	/** Give up (returning false) after this long; defaults to 30 s. */
	budgetMs?: number;
	/** Re-send `Target.closeTarget` this often while the target lingers; defaults to 2 s. */
	intervalMs?: number;
	/** Abort early (returning false). */
	signal?: AbortSignal;
}

/**
 * Close `targetId` in the shared browser and confirm through Chromium's own
 * `Target.getTargets` that it is gone, re-sending `Target.closeTarget` every
 * `intervalMs` until it is, `budgetMs` elapses, or the connection drops.
 *
 * Neither the reply to `Target.closeTarget` nor puppeteer's view is evidence:
 * Chromium can answer `success: true` and keep the page, puppeteer drops a
 * target from `browser.targets()` when its own session detaches, and
 * `page.close()` waits for the page's tab target, which may never close, while
 * Chromium may still list the page. An already-absent target costs one
 * `Target.getTargets` and no close. Never throws.
 */
export async function closeSharedTargetUntilGone(
	browser: Browser,
	targetId: string,
	opts: CloseUntilGoneOptions = {},
): Promise<boolean> {
	const intervalMs = opts.intervalMs ?? RECLOSE_INTERVAL_MS;
	const budget = AbortSignal.timeout(opts.budgetMs ?? RECLOSE_BUDGET_MS);
	const stop = opts.signal ? AbortSignal.any([budget, opts.signal]) : budget;
	// The root connection answers browser-level `Target.*` without attaching.
	// A browser-target session per call does not survive concurrency: puppeteer
	// tracks manual attaches in a Set keyed by target id
	// (`Connection.#manuallyAttached`, puppeteer-core/lib/puppeteer/cdp/Connection.js:221-237),
	// so of two overlapping attaches to the browser target the second is taken
	// for an auto-attach, and detaching it drops the browser target:
	// `browser.target()` then throws for the rest of the connection.
	let connection: Connection;
	try {
		connection = browserConnection(browser);
	} catch {
		return false;
	}
	let resend: AbortSignal | undefined;
	let pollMs = CLOSE_POLL_MIN_MS;
	// Every wait is an AbortSignal deadline, so the loop is bounded by `stop`
	// even when a send never answers. A closed connection rejects every send at
	// once (Connection.js:111-113), so `browser.connected`
	// (puppeteer-core/lib/puppeteer/cdp/Browser.js:362-364) ends the loop
	// instead of letting it spin to the budget.
	while (!stop.aborted && browser.connected) {
		const listing = AbortSignal.any([stop, AbortSignal.timeout(intervalMs)]);
		const gone = await untilAborted(listing, () => connection.send("Target.getTargets")).then(
			({ targetInfos }) => !targetInfos.some(info => info.targetId === targetId),
			() => false,
		);
		if (gone) return true;
		if (!resend || resend.aborted) {
			// The reply is ignored either way: success can leave the page open,
			// and an error may only mean it already went. A send that never
			// answers is abandoned at the next re-send deadline, not the budget.
			resend = AbortSignal.timeout(intervalMs);
			await untilAborted(AbortSignal.any([stop, resend]), () =>
				connection.send("Target.closeTarget", { targetId }),
			).catch(() => undefined);
			pollMs = CLOSE_POLL_MIN_MS;
			continue;
		}
		const poll = AbortSignal.any([stop, resend, AbortSignal.timeout(pollMs)]);
		await new Promise<void>(resolve => {
			if (poll.aborted) resolve();
			else poll.addEventListener("abort", () => resolve(), { once: true });
		});
		pollMs = Math.min(pollMs * 2, CLOSE_POLL_MAX_MS);
	}
	return false;
}

/** Per-target close bounds for the detached sweeps; the defaults keep a sweep bounded, the seam is for tests. */
export interface SharedTargetSweepOptions {
	/** Per-target {@link CloseUntilGoneOptions.budgetMs}; defaults to 5 s. */
	budgetMs?: number;
	/** Per-target {@link CloseUntilGoneOptions.intervalMs}; defaults to 2 s. */
	intervalMs?: number;
}

/**
 * Remember a shared-browser target this process failed to confirm closed, for
 * retry while the process lives: the orphan reaper skips a live owner's
 * records, so without this a long-lived process would keep the page open until
 * it exits. The ownership record stays.
 */
export function markSharedTargetUnclosed(scope: SharedTargetScope, targetId: string): void {
	const dir = registryDir(scope);
	let unclosed = unclosedByDir.get(dir);
	if (!unclosed) {
		unclosed = new Set();
		unclosedByDir.set(dir, unclosed);
	}
	unclosed.add(targetId);
}

/**
 * Retry closing every target marked unclosed for `scope` with
 * {@link closeSharedTargetUntilGone}; a confirmed close unmarks the target and
 * forgets its ownership record, an unconfirmed one stays marked for the next
 * retry. Returns the count closed. Never throws; concurrent calls for the same
 * scope share one run.
 */
export function retryUnclosedSharedTargets(
	browser: Browser,
	scope: SharedTargetScope,
	opts: SharedTargetSweepOptions = {},
): Promise<number> {
	const dir = registryDir(scope);
	const inFlight = unclosedRetries.get(dir);
	if (inFlight) return inFlight;
	const run = retryUnclosed(browser, scope, dir, opts).catch(err => {
		logger.debug("Failed to retry unclosed shared-browser targets", {
			error: err instanceof Error ? err.message : String(err),
		});
		return 0;
	});
	unclosedRetries.set(dir, run);
	void run.finally(() => {
		if (unclosedRetries.get(dir) === run) unclosedRetries.delete(dir);
	});
	return run;
}

async function retryUnclosed(
	browser: Browser,
	scope: SharedTargetScope,
	dir: string,
	opts: SharedTargetSweepOptions,
): Promise<number> {
	const pending = unclosedByDir.get(dir);
	if (!pending || pending.size === 0) return 0;
	let closed = 0;
	// Snapshot: a close confirmed below unmarks its id, and a search may mark new ones mid-sweep.
	for (const targetId of Array.from(pending)) {
		const gone = await closeSharedTargetUntilGone(browser, targetId, {
			budgetMs: opts.budgetMs ?? SWEEP_CLOSE_BUDGET_MS,
			intervalMs: opts.intervalMs,
		});
		if (!gone) {
			logger.debug("Shared-browser target still open after retried close", { targetId });
			continue;
		}
		closed++;
		await forgetSharedTarget(scope, targetId);
	}
	if (unclosedByDir.get(dir)?.size === 0) unclosedByDir.delete(dir);
	if (closed > 0) logger.debug("Closed lingering shared-browser targets", { count: closed, daemon: scope.daemonName });
	return closed;
}

/** Atomically retain unresolved targets, or remove an ownership file once all are resolved. */
async function updateOwnershipFile(owner: OrphanOwner, targetIds: string[]): Promise<void> {
	if (targetIds.length === 0) {
		await fs.rm(owner.file, { force: true });
		return;
	}
	const tmp = `${owner.file}.${process.pid}.tmp`;
	try {
		const record: OwnershipFile = { pid: owner.pid, updatedAt: owner.updatedAt, targets: targetIds };
		await Bun.write(tmp, JSON.stringify(record));
		await fs.rename(tmp, owner.file);
	} catch (error) {
		await fs.rm(tmp, { force: true }).catch(() => undefined);
		throw error;
	}
}

/**
 * Reap shared-browser targets whose owning omp process is gone. A target
 * counts as closed only once Chromium no longer lists it
 * ({@link closeSharedTargetUntilGone}, with a per-target sweep budget). Each
 * owner file is removed only after every target is confirmed gone; partial
 * failures atomically retain the unresolved ids for the next attach to retry.
 * Failures are logged, never thrown, so cleanup cannot block browser open.
 */
export async function reapOrphanSharedTargets(
	browser: Browser,
	scope: SharedTargetScope,
	opts: SharedTargetSweepOptions = {},
): Promise<number> {
	let scan: OrphanScan;
	try {
		scan = await collectOrphanTargets(scope);
	} catch (err) {
		logger.debug("Failed to scan shared-browser target registry", {
			error: err instanceof Error ? err.message : String(err),
		});
		return 0;
	}
	let closed = 0;
	for (const owner of scan.owners) {
		const retained: string[] = [];
		for (const targetId of owner.targetIds) {
			const gone = await closeSharedTargetUntilGone(browser, targetId, {
				budgetMs: opts.budgetMs ?? SWEEP_CLOSE_BUDGET_MS,
				intervalMs: opts.intervalMs,
			});
			if (gone) {
				closed++;
			} else {
				retained.push(targetId);
				logger.debug("Retaining orphaned shared-browser target for retry", { targetId, ownerPid: owner.pid });
			}
		}
		if (retained.length === owner.targetIds.length) continue;
		try {
			await updateOwnershipFile(owner, retained);
		} catch (err) {
			logger.debug("Failed to update shared-browser ownership after reap", {
				ownerPid: owner.pid,
				error: err instanceof Error ? err.message : String(err),
			});
		}
	}
	if (closed > 0) logger.debug("Reaped orphaned shared-browser targets", { count: closed, daemon: scope.daemonName });
	return closed;
}

/** Test-only reset of the in-process ownership state. */
export function resetOrphanRegistryForTest(): void {
	ownedByDir.clear();
	unclosedByDir.clear();
	unclosedRetries.clear();
	writeChains.clear();
}
