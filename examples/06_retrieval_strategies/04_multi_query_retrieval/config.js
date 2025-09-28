/**
 * Multi-query retrieval configuration.
 * Defaults can be overridden via environment variables.
 */

const DEFAULTS = {
    /** Vector dimension (must match embedding model) */
    dim: 384,
    /** Max elements in vector store */
    maxElements: 10_000,
    /** Namespace for this example's vectors */
    namespace: "multi_query",
    /** RRF constant k (higher = more weight on top ranks) */
    rrfK: 60,
    /** Max sub-queries for decomposition/expansion (avoid runaway) */
    maxSubQueries: 7,
    /** Top-k results per single query */
    topKPerQuery: 5,
    /** Timeout in ms for a single embed or search (0 = no timeout) */
    timeoutMs: 0,
    /** Retries for embed/search on failure */
    retries: 1,
    /** Max concurrent queries when running in parallel (0 = unlimited) */
    concurrency: 0,
    /** Log level: 'debug' | 'info' | 'warn' | 'error' */
    logLevel: "info",
};

const ENV_MAP = {
    MULTI_QUERY_DIM: "dim",
    MULTI_QUERY_MAX_ELEMENTS: "maxElements",
    MULTI_QUERY_NAMESPACE: "namespace",
    MULTI_QUERY_RRF_K: "rrfK",
    MULTI_QUERY_MAX_SUB_QUERIES: "maxSubQueries",
    MULTI_QUERY_TOP_K_PER_QUERY: "topKPerQuery",
    MULTI_QUERY_TIMEOUT_MS: "timeoutMs",
    MULTI_QUERY_RETRIES: "retries",
    MULTI_QUERY_CONCURRENCY: "concurrency",
    LOG_LEVEL: "logLevel",
};

function parseEnv() {
    const out = {};
    for (const [envKey, configKey] of Object.entries(ENV_MAP)) {
        const v = process.env[envKey];
        if (v === undefined || v === "") continue;
        if (configKey === "logLevel") {
            out[configKey] = v;
            continue;
        }
        const n = Number(v);
        if (!Number.isNaN(n)) out[configKey] = n;
    }
    return out;
}

/**
 * @returns {typeof DEFAULTS} Config object with env overrides applied.
 */
export function getConfig() {
    const env = parseEnv();
    return { ...DEFAULTS, ...env };
}

export const config = getConfig();
