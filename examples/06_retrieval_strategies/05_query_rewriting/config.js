/**
 * Query rewriting configuration.
 * Defaults can be overridden via environment variables.
 */

/** Default acronym/term expansions for heuristic rewrite */
const DEFAULT_ACRONYM_MAP = {
    ml: "machine learning",
    ai: "artificial intelligence",
    js: "javascript",
    py: "python",
    db: "database",
    api: "API",
    rest: "REST",
    graphql: "GraphQL",
    abt: "about",
    cicd: "CI/CD",
    iac: "infrastructure as code",
    k8s: "kubernetes",
    oss: "open source",
    docs: "documentation",
    repo: "repository",
    diff: "difference",
    vs: "versus",
    e2e: "end to end",
    tdd: "test driven development",
};

const DEFAULTS = {
    /** Max length (chars) for rewritten query */
    maxRewriteLength: 300,
    /** Max alternate queries to produce (0 = none) */
    maxAlternates: 4,
    /** Allowed filter keys for structured constraints (e.g. category, date) */
    allowedFilterKeys: ["category", "topic", "dateFrom", "dateTo"],
    /** Use LLM for rewrite when available (fallback to heuristic if false or LLM fails) */
    useLLM: true,
    /** Path to LLM GGUF for query rewriting (Qwen 3-1; set to empty to skip LLM load) */
    llmModelPath: "",
    /** Timeout in ms for LLM call (0 = no timeout) */
    llmTimeoutMs: 10_000,
    /** Filler phrases to strip from user query */
    stripPhrases: ["please", "can you", "could you", "would you", "tell me", "i need", "i want", "hey", "thanks", "thank you", "plz", "pls", "thx"],
    /** Log level */
    logLevel: "info",
};

const ENV_MAP = {
    QUERY_REWRITE_MAX_LENGTH: "maxRewriteLength",
    QUERY_REWRITE_MAX_ALTERNATES: "maxAlternates",
    QUERY_REWRITE_USE_LLM: "useLLM",
    QUERY_REWRITE_LLM_MODEL_PATH: "llmModelPath",
    QUERY_REWRITE_LLM_TIMEOUT_MS: "llmTimeoutMs",
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
        if (configKey === "useLLM") {
            out[configKey] = v === "1" || v.toLowerCase() === "true";
            continue;
        }
        if (configKey === "llmModelPath") {
            out[configKey] = v;
            continue;
        }
        const n = Number(v);
        if (!Number.isNaN(n)) out[configKey] = n;
    }
    return out;
}

export function getConfig() {
    const env = parseEnv();
    return {
        ...DEFAULTS,
        ...env,
        acronymMap: { ...DEFAULT_ACRONYM_MAP },
    };
}

export const config = getConfig();
export { DEFAULT_ACRONYM_MAP };
