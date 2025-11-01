/**
 * Simple logger with levels and optional timing for observability.
 */

const LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };

/**
 * @param {string} level 'debug' | 'info' | 'warn' | 'error'
 * @param {object} [config] Optional { logLevel } (default from process.env LOG_LEVEL or 'info')
 */
export function createLogger(config = {}) {
    const logLevel = (config.logLevel ?? process.env.LOG_LEVEL ?? "info").toLowerCase();
    const levelIndex = LEVELS[logLevel] ?? LEVELS.info;

    const should = (level) => (LEVELS[level] ?? 0) >= levelIndex;

    return {
        debug(...args) {
            if (should("debug")) console.log("[multi-query][debug]", ...args);
        },
        info(...args) {
            if (should("info")) console.log("[multi-query][info]", ...args);
        },
        warn(...args) {
            if (should("warn")) console.warn("[multi-query][warn]", ...args);
        },
        error(...args) {
            if (should("error")) console.error("[multi-query][error]", ...args);
        },
        /**
         * Time an async operation and log duration at info level.
         * @template T
         * @param {string} label
         * @param {() => Promise<T>} fn
         * @returns {Promise<T>}
         */
        async timeAsync(label, fn) {
            const start = Date.now();
            try {
                const result = await fn();
                if (should("info")) {
                    console.log(`[multi-query][info] ${label} ${Date.now() - start}ms`);
                }
                return result;
            } catch (e) {
                if (should("error")) {
                    console.error(`[multi-query][error] ${label} failed after ${Date.now() - start}ms`, e?.message ?? e);
                }
                throw e;
            }
        },
        /**
         * Return metrics for a retrieval run (for logging or monitoring).
         * @param {{ queryCount: number; totalResults: number; uniqueAfterDedup: number; durationMs: number }} m
         */
        metrics(m) {
            if (should("info")) {
                console.log("[multi-query][metrics]", m);
            }
        },
    };
}

export const logger = createLogger();
