/**
 * Regression tests for web_search leaking pages into the project-shared
 * headless Chromium (`omp.browser.headless`). One long-lived session piled up
 * ~50 pages (about:blank, Cloudflare challenges, 403s) and 13–18 GB RSS:
 *
 * - public engines are aborted mid-flight after their soft deadline, and an
 *   abort during `newPage()` dropped the page puppeteer was still creating —
 *   its target already existed, so it stayed open as about:blank;
 * - a failed or hung `page.close()` was swallowed, with no retry by target id;
 * - search pages were never claimed in the shared-target ownership registry,
 *   so the orphan reaper could not close them after their owner died.
 *
 * The fake Chromium answers browser-level `Target.*` on its root connection
 * and keeps its own target list (`Target.getTargets`); each test asserts on
 * that list and on the durable ownership record. Its `page.close()` never
 * settles, as puppeteer's does on a connection that outlives a search.
 */

import { afterEach, describe, expect, it, spyOn, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { logger, withTimeout } from "@oh-my-pi/pi-utils";
import { daemonRuntimeDir } from "@oh-my-pi/pi-coding-agent/launch/paths";
import * as launch from "@oh-my-pi/pi-coding-agent/tools/browser/launch";
import * as orphanRegistry from "@oh-my-pi/pi-coding-agent/tools/browser/orphan-registry";
import {
	forgetSharedTarget,
	recordSharedTarget,
	resetOrphanRegistryForTest,
	type SharedTargetScope,
} from "@oh-my-pi/pi-coding-agent/tools/browser/orphan-registry";
import * as registry from "@oh-my-pi/pi-coding-agent/tools/browser/registry";
import type { PuppeteerBrowserHandle } from "@oh-my-pi/pi-coding-agent/tools/browser/registry";
import * as sharedDaemon from "@oh-my-pi/pi-coding-agent/tools/browser/shared-daemon";
import { browserFetch } from "@oh-my-pi/pi-coding-agent/web/search/providers/browser-page";
import type { Page } from "puppeteer-core";

const DAEMON_NAME = "omp.browser.headless";
const SEARCH_URL = "https://search.example/html?q=leak";
const FINAL_URL = "https://search.example/html?q=leak&page=1";
const RESULT_HTML = "<html><body><a class='result'>hit</a></body></html>";
/** Target id of the first page the fake Chromium opens. */
const FIRST_TARGET = "TARGET-1";

const scopes: SharedTargetScope[] = [];

/** Unique per-test scope; `afterEach` removes its registry dir even when a test fails. */
function trackedScope(): SharedTargetScope {
	const scope = {
		projectDir: path.join("/tmp", `omp-search-leak-test-${crypto.randomUUID()}`),
		daemonName: DAEMON_NAME,
	};
	scopes.push(scope);
	return scope;
}

/** Targets this process still claims ownership of, straight from the durable record. */
async function ownedTargets(scope: SharedTargetScope): Promise<string[]> {
	// Ownership writes are fire-and-forget; a record→forget pair for a
	// throwaway id joins the registry's write chain, so the file read below
	// reflects every earlier write.
	const probe = `chain-probe-${crypto.randomUUID()}`;
	await recordSharedTarget(scope, probe);
	await forgetSharedTarget(scope, probe);
	const file = path.join(daemonRuntimeDir(scope.projectDir), `${scope.daemonName}.targets`, `${process.pid}.json`);
	const raw = await Bun.file(file)
		.text()
		.catch(() => null);
	if (raw === null) return [];
	const parsed = JSON.parse(raw) as { targets?: unknown };
	return Array.isArray(parsed.targets) ? parsed.targets.filter((id): id is string => typeof id === "string") : [];
}

/**
 * Browser-level `Target.closeTarget`: closes the target, or answers
 * `success: true` while the page lingers, as real Chromium does for pages
 * aborted mid-navigation.
 */
type CloseById = "ok" | "linger";

interface FakeChromiumOptions {
	closeById?: CloseById;
	/** Holds `newPage()` after its target exists, as puppeteer does while it attaches. */
	newPageGate?: Promise<void>;
	/** Runs inside every navigation, while the search page is open. */
	onGoto?: () => Promise<void>;
	/** Runs at the start of every browser-level `Target.closeTarget`, before it takes effect. */
	onCloseById?: () => Promise<void>;
	/** Keep the real cleanup deadlines instead of scaling them down. */
	realTimeouts?: boolean;
}

interface FakeChromium {
	handle: PuppeteerBrowserHandle;
	/** Targets Chromium still has open, as `Target.getTargets` reports them. */
	open: Set<string>;
	/** Target ids a browser-level `Target.closeTarget` was sent for. */
	closeByIdRequests: string[];
	/** Browser-level `Target.closeTarget` behavior; tests may change it mid-run. */
	closeById: CloseById;
	/** Resolves once `newPage()` has created its target. */
	pageCreated: Promise<void>;
	/** Resolves when the last browser hold is released (the shared handle disconnects). */
	disconnected: Promise<void>;
	/** Connect again for a later search, re-arming {@link FakeChromium.disconnected}. */
	reconnect(): void;
	/** The CDP connection drops (Chromium died): puppeteer reports the browser disconnected. */
	loseConnection(): void;
}

function fakeChromium(scope: SharedTargetScope, options: FakeChromiumOptions = {}): FakeChromium {
	const { newPageGate, onGoto, onCloseById } = options;
	const open = new Set<string>();
	const pageCreated = Promise.withResolvers<void>();
	let disconnected = Promise.withResolvers<void>();
	let nextTarget = 0;

	const makePage = (targetId: string): Page =>
		({
			target: () => ({ _targetId: targetId }),
			goto: async () => {
				await onGoto?.();
				return { status: () => 200 };
			},
			waitForSelector: async () => null,
			content: async () => RESULT_HTML,
			url: () => FINAL_URL,
			// Puppeteer's `page.close()` sends `Target.closeTarget`, then waits for
			// the page's tab target to detach. A page first attached through a
			// browser-target session has the browser target as its "tab", so on a
			// connection that outlives the search that wait never ends.
			close: (): Promise<void> => new Promise<void>(() => {}),
		}) as unknown as Page;

	/** Root CDP connection (`browserConnection`), where browser-level `Target.*` commands go. */
	const rootConnection = {
		send: async (method: string, params?: { targetId?: string }) => {
			if (method === "Target.closeTarget") {
				const targetId = params?.targetId ?? "";
				chromium.closeByIdRequests.push(targetId);
				await onCloseById?.();
				if (chromium.closeById === "ok") open.delete(targetId);
				return { success: true };
			}
			if (method === "Target.getTargets") {
				return { targetInfos: [...open].map(targetId => ({ targetId, type: "page" })) };
			}
			throw new Error(`unexpected CDP method ${method}`);
		},
		createSession: async () => {
			throw new Error("page cleanup must not attach to targets");
		},
	};

	const browser = {
		connected: true,
		newPage: (): Promise<Page> => {
			// Puppeteer's `newPage()` sends `Target.createTarget` first, so the
			// page exists in Chromium before the promise settles.
			const targetId = `TARGET-${++nextTarget}`;
			open.add(targetId);
			pageCreated.resolve();
			const page = makePage(targetId);
			return newPageGate ? newPageGate.then(() => page) : Promise.resolve(page);
		},
		_connection: rootConnection,
		// Concurrent browser-target sessions break puppeteer's connection; cleanup must not attach.
		target: () => ({
			createCDPSession: async () => {
				throw new Error("page cleanup must not attach to the browser target");
			},
		}),
		disconnect: () => {
			browser.connected = false;
			disconnected.resolve();
		},
	};

	const handle = {
		key: `headless:search-leak-${crypto.randomUUID()}`,
		kind: { kind: "headless", headless: true },
		refCount: 0,
		browser,
		sharedDaemon: { name: scope.daemonName, projectDir: scope.projectDir },
		stealth: { browserSession: null, override: null },
	} as unknown as PuppeteerBrowserHandle;

	spyOn(registry, "acquireBrowser").mockResolvedValue(handle);
	spyOn(launch, "applyViewport").mockResolvedValue(undefined);
	spyOn(launch, "applyStealthPatches").mockResolvedValue(undefined);
	// The fake scope has no broker behind it; a stop request must never reach one.
	spyOn(sharedDaemon, "stopSharedBrowserIfUnreachable").mockResolvedValue(false);
	// The plain fetch is blocked, so `browserFetch` escalates to the browser.
	spyOn(globalThis, "fetch").mockResolvedValue(new Response("blocked", { status: 403 }));
	if (!options.realTimeouts) {
		// Scale the cleanup deadlines (5 s close cap, 2 s re-close interval, 30 s
		// re-close budget, 90 s abandoned-page wait) down 250x, keeping their
		// order. Only the page cleanup uses `AbortSignal.timeout` on this path.
		const realTimeout = AbortSignal.timeout.bind(AbortSignal);
		spyOn(AbortSignal, "timeout").mockImplementation(ms => realTimeout(ms / 250));
	}

	const chromium: FakeChromium = {
		handle,
		open,
		closeByIdRequests: [],
		closeById: options.closeById ?? "ok",
		pageCreated: pageCreated.promise,
		get disconnected() {
			return disconnected.promise;
		},
		reconnect() {
			browser.connected = true;
			disconnected = Promise.withResolvers<void>();
		},
		loseConnection() {
			browser.connected = false;
			disconnected.resolve();
		},
	};
	return chromium;
}

function search(signal: AbortSignal = new AbortController().signal) {
	return browserFetch(SEARCH_URL, { signal, browser: { shouldFallback: () => false } });
}

/** Wait for background page cleanup to drop its browser hold. */
async function cleanupFinished(chromium: FakeChromium): Promise<void> {
	// Only bounds a failing test; a passing one resolves as soon as cleanup releases the browser.
	await withTimeout(chromium.disconnected, 10_000, "web search page cleanup never released the browser");
}

afterEach(async () => {
	vi.restoreAllMocks();
	resetOrphanRegistryForTest();
	for (const scope of scopes.splice(0)) {
		await fs.rm(daemonRuntimeDir(scope.projectDir), { recursive: true, force: true }).catch(() => undefined);
	}
});

describe("web search browser page cleanup in the shared Chromium", () => {
	it("claims and closes the page of a newPage() that settles after the search was aborted", async () => {
		const scope = trackedScope();
		const gate = Promise.withResolvers<void>();
		let ownedWhileClosing: string[] = [];
		// Real deadlines: the fake's close answer awaits a disk read, which under a
		// scaled-down 8 ms re-send interval could be re-sent on a loaded machine.
		const chromium = fakeChromium(scope, {
			realTimeouts: true,
			newPageGate: gate.promise,
			onCloseById: async () => {
				ownedWhileClosing = await ownedTargets(scope);
			},
		});
		const controller = new AbortController();

		const searching = search(controller.signal);
		await chromium.pageCreated;
		controller.abort(new Error("soft deadline"));
		await expect(searching).rejects.toMatchObject({ name: "AbortError" });

		// The caller is gone but its about:blank target is not: the shared
		// browser must stay connected so the late page can still be closed.
		expect(chromium.handle.browser.connected).toBe(true);
		gate.resolve();
		await cleanupFinished(chromium);

		// Claimed before the close starts, so the reaper finds it if this process dies mid-close.
		expect(ownedWhileClosing).toEqual([FIRST_TARGET]);
		expect(chromium.closeByIdRequests).toEqual([FIRST_TARGET]);
		expect([...chromium.open]).toEqual([]);
		expect(await ownedTargets(scope)).toEqual([]);
	});

	it("returns without waiting out the close deadline when puppeteer's page.close() would never settle", async () => {
		const scope = trackedScope();
		// Real deadlines: a teardown waiting on `page.close()` would take the full 5 s cap.
		const chromium = fakeChromium(scope, { realTimeouts: true });

		const started = performance.now();
		await search();
		const elapsedMs = performance.now() - started;
		await cleanupFinished(chromium);

		expect(elapsedMs).toBeLessThan(1_000);
		expect([...chromium.open]).toEqual([]);
		expect(await ownedTargets(scope)).toEqual([]);
	});

	it("keeps the ownership record, marks it for retry, and warns when Chromium keeps the page open", async () => {
		const scope = trackedScope();
		const chromium = fakeChromium(scope, { closeById: "linger" });
		const warn = spyOn(logger, "warn");
		const markUnclosed = spyOn(orphanRegistry, "markSharedTargetUnclosed");

		await search();
		await cleanupFinished(chromium);

		// Still open in Chromium, so it must stay claimed for the orphan reaper.
		expect([...chromium.open]).toEqual([FIRST_TARGET]);
		expect(await ownedTargets(scope)).toEqual([FIRST_TARGET]);
		expect(warn).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ targetId: FIRST_TARGET }));
		expect(markUnclosed).toHaveBeenCalledWith(scope, FIRST_TARGET);
		// One stuck page must not take down the browser other sessions' tabs live in.
		expect(sharedDaemon.stopSharedBrowserIfUnreachable).not.toHaveBeenCalled();
	});

	it("keeps the page claimed for retry without warning when the browser connection is lost", async () => {
		const scope = trackedScope();
		const chromium = fakeChromium(scope, {
			onGoto: async () => {
				chromium.loseConnection();
				throw new Error("Protocol error: Connection closed.");
			},
		});
		const warn = spyOn(logger, "warn");
		const debug = spyOn(logger, "debug");
		const markUnclosed = spyOn(orphanRegistry, "markSharedTargetUnclosed");

		await expect(search()).rejects.toThrow("Connection closed");
		await cleanupFinished(chromium);

		// The page most likely died with the browser; the next attach's retry confirms it.
		expect(await ownedTargets(scope)).toEqual([FIRST_TARGET]);
		expect(markUnclosed).toHaveBeenCalledWith(scope, FIRST_TARGET);
		expect(debug).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ targetId: FIRST_TARGET }));
		expect(warn).not.toHaveBeenCalled();
	});

	it("closes a page an earlier search left open when a later search acquires the browser", async () => {
		const scope = trackedScope();
		const chromium = fakeChromium(scope, { closeById: "linger" });

		await search();
		await cleanupFinished(chromium);
		expect([...chromium.open]).toEqual([FIRST_TARGET]);
		expect(await ownedTargets(scope)).toEqual([FIRST_TARGET]);

		// The owner is alive, so the orphan reaper skips its page; the next
		// search must retry the close itself.
		chromium.closeById = "ok";
		chromium.reconnect();
		await search();
		await cleanupFinished(chromium);

		expect([...chromium.open]).toEqual([]);
		expect(await ownedTargets(scope)).toEqual([]);
	});

	it("returns the loaded page, claims it while open, and closes it once", async () => {
		const scope = trackedScope();
		let ownedDuringSearch: string[] = [];
		// Real deadlines, so a slow machine cannot turn one close into a re-send.
		const chromium = fakeChromium(scope, {
			realTimeouts: true,
			onGoto: async () => {
				ownedDuringSearch = await ownedTargets(scope);
			},
		});

		const loaded = await search();

		expect(loaded).toEqual({ html: RESULT_HTML, status: 200, url: FINAL_URL });
		expect(ownedDuringSearch).toEqual([FIRST_TARGET]);
		expect(chromium.closeByIdRequests).toEqual([FIRST_TARGET]);
		expect([...chromium.open]).toEqual([]);
		expect(chromium.handle.browser.connected).toBe(false);
		expect(await ownedTargets(scope)).toEqual([]);
	});
});
