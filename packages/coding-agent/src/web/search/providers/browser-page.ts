import type { FetchImpl } from "@oh-my-pi/pi-ai";
import { getProjectDir, logger, untilAborted } from "@oh-my-pi/pi-utils";
import type { Page, Target } from "puppeteer-core";
import { applyStealthPatches, applyViewport, BROWSER_PROTOCOL_TIMEOUT_MS } from "../../../tools/browser/launch";
import {
	closeSharedTargetUntilGone,
	forgetSharedTarget,
	markSharedTargetUnclosed,
	recordSharedTarget,
	retryUnclosedSharedTargets,
	type SharedTargetScope,
} from "../../../tools/browser/orphan-registry";
import {
	acquireBrowser,
	holdBrowser,
	type PuppeteerBrowserHandle,
	releaseBrowser,
	sharedTargetScopeOf,
} from "../../../tools/browser/registry";
import { buildBrowserNavigationHeaders } from "./browser-headers";
import { SEARCH_HARD_TIMEOUT_MS } from "./utils";

/** HTML plus the response status and final URL after redirects or browser navigation. */
export interface LoadedHtmlPage {
	html: string;
	status: number;
	url: string;
}

interface BrowserFallbackOptions {
	homeUrl?: string;
	ready?: { selector: string; timeoutMs: number };
	afterNavigation?: (page: Page, signal: AbortSignal) => Promise<void>;
	shouldFallback: (page: LoadedHtmlPage) => boolean;
	attempts?: number;
	retryDelayMs?: number;
}

/** Controls a browser-profiled fetch and its optional headless-browser fallback. */
export interface BrowserFetchOptions {
	fetch?: FetchImpl;
	signal: AbortSignal;
	timeoutMs?: number;
	randomizeHeaders?: boolean;
	referer?: string;
	init?: Omit<RequestInit, "headers" | "signal">;
	headers?: Readonly<Record<string, string>>;
	browser?: BrowserFallbackOptions;
}

/**
 * How long a search's teardown waits for its page close; a close still
 * unconfirmed by then finishes in the background. A wedged connection can
 * leave CDP calls pending forever, and `.catch()` only covers rejection, not
 * a hang, so cleanup needs its own deadline (issue #8865).
 */
const PAGE_CLOSE_TIMEOUT_MS = 5_000;

/**
 * How long cleanup waits for a `newPage()` whose caller aborted. Puppeteer
 * sends `Target.createTarget` before it resolves, so the about:blank page
 * already exists in the shared Chromium and only the late `Page` can close
 * it. `newPage()` can take up to the `Target.createTarget` protocol timeout
 * plus puppeteer's default 30 s `waitForTarget` for the new target
 * (node_modules/puppeteer-core/lib/puppeteer/cdp/Browser.js:208-222,
 * lib/puppeteer/api/Browser.js:92); this cap only keeps the cleanup's browser
 * hold from outliving a wedged connection.
 */
const ABANDONED_PAGE_TIMEOUT_MS = BROWSER_PROTOCOL_TIMEOUT_MS + 30_000;

/** CDP target id of a page target; puppeteer keeps it private (`_targetId`), as the browser tool reads it. */
function targetIdOf(target: Target): string | undefined {
	const raw = target as unknown as { _targetId?: unknown };
	return typeof raw._targetId === "string" ? raw._targetId : undefined;
}

/** Claim a search page in the shared-target registry so it is reaped if this process dies before closing it (issue #10022). */
function recordSearchPage(handle: PuppeteerBrowserHandle, page: Page): void {
	const scope = sharedTargetScopeOf(handle);
	const targetId = targetIdOf(page.target());
	if (scope && targetId) void recordSharedTarget(scope, targetId);
}

/**
 * Close a search page and settle its ownership record. Never rejects. The
 * record is dropped only once Chromium itself no longer lists the target; an
 * unconfirmed page stays claimed and is retried by this process's later
 * searches (or reaped after it dies).
 *
 * Closes by target id instead of `page.close()`, which sends the same
 * `Target.closeTarget` on the root connection and then awaits the closed
 * deferred of what puppeteer takes for the page's tab target
 * (puppeteer-core/lib/puppeteer/cdp/Page.js:938-941). On a connection that
 * outlives a search, pages are often first attached through a browser-target
 * session (`applyStealthPatches` opens one per concurrent first call, and
 * `Connection.#manuallyAttached`, Connection.js:228-236, lets TargetManager
 * auto-attach through all but one), so their "tab" is the browser target
 * (Page.js:150-154) and that close never settles while the connection lives.
 * Chromium's detach/destroy events clear puppeteer's sessions and targets
 * either way (Connection.js:156-166, TargetManager.js:194-197, 340-351).
 */
async function closeSearchPage(handle: PuppeteerBrowserHandle, page: Page): Promise<void> {
	const targetId = targetIdOf(page.target());
	if (!targetId) {
		const pageClosed = await untilAborted(AbortSignal.timeout(PAGE_CLOSE_TIMEOUT_MS), () => page.close()).then(
			() => true,
			() => false,
		);
		if (!pageClosed) logger.warn("Could not close a web search page; it may still be open");
		return;
	}
	const closed = await closeSharedTargetUntilGone(handle.browser, targetId);
	if (!closed) {
		// Without a connection the page most likely died with the browser; the
		// retry on the next attach confirms that against the new connection.
		if (handle.browser.connected) {
			logger.warn("Could not confirm a web search page was closed; it may still be open", { targetId });
		} else {
			logger.debug("Lost the browser connection before a web search page close was confirmed", { targetId });
		}
	}
	const scope = sharedTargetScopeOf(handle);
	if (!scope) return;
	if (closed) {
		void forgetSharedTarget(scope, targetId);
		return;
	}
	// Keep the ownership record and retry while this process lives.
	markSharedTargetUnclosed(scope, targetId);
}

/**
 * {@link closeSearchPage} under its own browser hold, so a caller that stops
 * waiting does not abandon the close: the last `releaseBrowser` disconnects
 * from the shared Chromium, and a page left mid-close stays open there.
 */
function closeSearchPageHeld(handle: PuppeteerBrowserHandle, page: Page): Promise<void> {
	holdBrowser(handle);
	return closeSearchPage(handle, page).finally(() => releaseBrowser(handle, { kill: false }).catch(() => undefined));
}

/** Close the page of a `newPage()` its caller stopped waiting for, once it settles. Fire-and-forget. */
function closeAbandonedPage(handle: PuppeteerBrowserHandle, pending: Promise<Page>): void {
	holdBrowser(handle);
	void untilAborted(AbortSignal.timeout(ABANDONED_PAGE_TIMEOUT_MS), pending)
		.then(
			page => {
				recordSearchPage(handle, page);
				return closeSearchPage(handle, page);
			},
			(error: unknown) =>
				// No `Page`, so no target id to close or claim by: a target
				// Chromium created before `newPage()` failed stays open.
				logger.debug("Could not resolve an abandoned web search page; its target may still be open", {
					error: error instanceof Error ? error.message : String(error),
				}),
		)
		.finally(() => releaseBrowser(handle, { kill: false }).catch(() => undefined));
}

/**
 * Retry closes this process could not confirm in earlier searches, under its
 * own browser hold so the retry outlives the search that started it. The
 * orphan reaper skips targets of live processes, so a long-lived session
 * would otherwise keep them open. Fire-and-forget.
 */
function retryUnclosedSearchPages(handle: PuppeteerBrowserHandle, scope: SharedTargetScope): void {
	holdBrowser(handle);
	void retryUnclosedSharedTargets(handle.browser, scope).finally(() =>
		releaseBrowser(handle, { kill: false }).catch(() => undefined),
	);
}

async function fetchHtmlPage(url: string, options: BrowserFetchOptions, fetchImpl: FetchImpl): Promise<LoadedHtmlPage> {
	const response = await fetchImpl(url, {
		...options.init,
		headers: {
			...buildBrowserNavigationHeaders({ randomized: options.randomizeHeaders }),
			...(options.referer ? { Referer: options.referer, "Sec-Fetch-Site": "same-origin" } : {}),
			...options.headers,
		},
		signal: options.signal,
	});
	return { html: await response.text(), status: response.status, url: response.url || url };
}

async function browseHtmlPage(
	url: string,
	options: BrowserFallbackOptions,
	signal: AbortSignal,
	timeoutMs = SEARCH_HARD_TIMEOUT_MS,
): Promise<LoadedHtmlPage> {
	const { homeUrl, ready } = options;
	const attempts = Math.max(1, options.attempts ?? 1);
	const handle = await untilAborted(signal, () =>
		acquireBrowser(
			{ kind: "headless", headless: true },
			{
				cwd: getProjectDir(),
				signal,
			},
		),
	);
	if (!("browser" in handle)) {
		await releaseBrowser(handle, { kill: false });
		throw new Error("Headless browser acquisition returned a non-Puppeteer browser");
	}
	const scope = sharedTargetScopeOf(handle);
	if (scope) retryUnclosedSearchPages(handle, scope);

	holdBrowser(handle);
	let page: Page | undefined;
	try {
		// Keep the pending page: `untilAborted` only stops waiting on abort,
		// it does not cancel `newPage()`, whose target already exists by the
		// time a mid-flight abort lands (public search engines are aborted
		// after their soft deadline routinely).
		const pending = handle.browser.newPage();
		let activePage: Page;
		try {
			activePage = await untilAborted(signal, pending);
		} catch (error) {
			if (signal.aborted) closeAbandonedPage(handle, pending);
			throw error;
		}
		page = activePage;
		recordSearchPage(handle, activePage);
		// Viewport and stealth setup talk to the same CDP session as the
		// navigations below but were previously awaited raw. When the shared
		// daemon or the page's target dies mid-setup, those puppeteer calls
		// never settle and the provider promise hangs past
		// SEARCH_HARD_TIMEOUT_MS — the abort signal had no listener at these
		// await points (issue #8865).
		await untilAborted(signal, () => applyViewport(activePage));
		await untilAborted(signal, () => applyStealthPatches(handle.browser, activePage, handle.stealth));
		if (homeUrl) {
			await untilAborted(signal, () =>
				activePage.goto(homeUrl, { waitUntil: "domcontentloaded", timeout: timeoutMs }),
			);
		}
		for (let attempt = 0; attempt < attempts; attempt++) {
			if (attempt > 0 && options.retryDelayMs) await Bun.sleep(options.retryDelayMs);

			const response = await untilAborted(signal, () =>
				activePage.goto(url, { waitUntil: "domcontentloaded", timeout: timeoutMs }),
			);
			if (options.afterNavigation) await options.afterNavigation(activePage, signal);
			if (ready) {
				await untilAborted(signal, () =>
					activePage.waitForSelector(ready.selector, { timeout: ready.timeoutMs }).catch(() => null),
				);
			}
			const loaded = {
				html: await untilAborted(signal, () => activePage.content()),
				status: response?.status() ?? 200,
				url: activePage.url(),
			};
			if (!options.shouldFallback(loaded) || attempt === attempts - 1) return loaded;
		}
		throw new Error("Browser fallback exhausted without a response");
	} finally {
		// Teardown must complete even when the caller's signal already fired
		// (a wedged connection leaves CDP calls pending), so bound it with a
		// fresh deadline instead of reusing `signal` (issue #8865). A close
		// still unconfirmed by then finishes in the background.
		if (page) {
			const closing = closeSearchPageHeld(handle, page);
			await untilAborted(AbortSignal.timeout(PAGE_CLOSE_TIMEOUT_MS), closing).catch(() => undefined);
		}
		await releaseBrowser(handle, { kill: false });
	}
}

/** Fetch with a fresh browser profile, escalating rejected production responses to the stealth browser. */
export async function browserFetch(url: string, options: BrowserFetchOptions): Promise<LoadedHtmlPage> {
	const fetchImpl = options.fetch ?? fetch;
	let page: LoadedHtmlPage;
	try {
		page = await fetchHtmlPage(url, options, fetchImpl);
	} catch (error) {
		if (options.fetch || !options.browser) throw error;
		return browseHtmlPage(url, options.browser, options.signal, options.timeoutMs);
	}

	if (!options.browser || options.fetch) return page;
	const isSuccessful = page.status >= 200 && page.status < 300;
	if (isSuccessful && !options.browser.shouldFallback(page)) return page;
	return browseHtmlPage(url, options.browser, options.signal, options.timeoutMs);
}
