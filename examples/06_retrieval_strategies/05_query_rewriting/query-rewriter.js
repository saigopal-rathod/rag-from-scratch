/**
 * Query rewriter: normalize, heuristic rewrite, optional LLM rewrite with fallback.
 * Framework-agnostic; no actions. Output: rewrittenQuery, alternateQueries, filters, diagnostics.
 */

/**
 * Normalize raw input for consistent search-friendly form.
 * - Collapse whitespace to single space
 * - Collapse repeated punctuation (??? -> ?, !! -> !, ... -> ., ,,, -> ,)
 * - Strip invisible/control characters and BOM
 * - Normalize smart quotes to straight quotes
 * @param {string} query
 * @returns {string}
 */
export function normalize(query) {
    if (typeof query !== "string") return "";
    let s = query
        .replace(/[\u200B-\u200D\uFEFF\u0000-\u001F\u007F-\u009F]/g, "")
        .replace(/\s+/g, " ")
        .trim();
    s = s
        .replace(/\?+/g, "?")
        .replace(/!+/g, "!")
        .replace(/\.{2,}/g, ".")
        .replace(/,+/g, ",")
        .replace(/[\u201C\u201D]/g, '"')
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/\s+/g, " ")
        .trim();
    return s;
}

/**
 * Classify query intent (heuristic).
 * @param {string} query
 * @returns {'factual'|'howto'|'comparison'|'troubleshooting'|'find_doc'}
 */
export function classifyIntent(query) {
    const q = query.toLowerCase();
    if (/\bvs\.?\b|\bversus\b|\bdifference between\b|\bcompare\b|\bcomparison\b/.test(q)) return "comparison";
    if (/\berror\b|\bnot working\b|\bfix\b|\bdebug\b|\bissue\b|\bproblem\b/.test(q)) return "troubleshooting";
    if (/\bhow (do|to|can)\b|\bhow do i\b|\bstep(s)?\b|\btutorial\b|\bguide\b/.test(q)) return "howto";
    if (/\bfind\b|\bsearch\b|\bwhere (is|can i)\b|\blocate\b/.test(q)) return "find_doc";
    return "factual";
}

/**
 * Strip common filler and politeness phrases (lowercased).
 * @param {string} query
 * @param {string[]} stripPhrases
 * @returns {{ cleaned: string; removed: string[] }}
 */
export function stripFiller(query, stripPhrases = []) {
    const removed = [];
    let cleaned = query.toLowerCase();
    const phrases = [...stripPhrases].sort((a, b) => b.length - a.length);
    for (const phrase of phrases) {
        const re = new RegExp(`\\b${escapeRegex(phrase)}\\b[,.]?\\s*`, "gi");
        if (re.test(cleaned)) {
            removed.push(phrase);
            cleaned = cleaned.replace(re, " ");
        }
    }
    cleaned = cleaned.replace(/\s+/g, " ").trim();
    return { cleaned, removed };
}

function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Expand acronyms and common abbreviations (word-boundary).
 * @param {string} query
 * @param {Record<string, string>} acronymMap
 * @returns {string}
 */
export function expandAcronyms(query, acronymMap = {}) {
    let out = query;
    for (const [abbr, full] of Object.entries(acronymMap)) {
        const re = new RegExp(`\\b${escapeRegex(abbr)}\\b`, "gi");
        out = out.replace(re, full);
    }
    return out.replace(/\s+/g, " ").trim();
}

/**
 * Strip content that looks like prompt injection (instructions to the system).
 * Keeps the user's actual question: removes "Ignore previous instructions", "say \"...\"",
 * role hijacking, and trailing instruction clauses so the result is just the question.
 * @param {string} query
 * @returns {{ cleaned: string; stripped: string[] }}
 */
export function stripInjection(query) {
    const stripped = [];
    const injectionPatterns = [
        /\b(ignore|forget|disregard)\s+(previous|above|all)\s+(instructions?|prompts?|context)\b/gi,
        /\byou\s+are\s+(now|a)\s+/gi,
        /\b(system|assistant|user):\s*/gi,
        /\b(respond|answer|output)\s+only\s+/gi,
        /\[INST\]|\[\/INST\]|<<SYS>>|<<\/SYS>>/gi,
        /\s+and\s+say\s+["'][^"']*["']\s*\.?/gi,
        /\s+and\s+(respond|answer|output)\s+(only\s+)?(with\s+)?["']?[^."']*["']?\s*\.?/gi,
    ];
    let cleaned = query;
    for (const re of injectionPatterns) {
        const before = cleaned;
        cleaned = cleaned.replace(re, " ");
        if (cleaned !== before) stripped.push(re.source);
    }
    cleaned = cleaned.replace(/\s+/g, " ").trim();
    return { cleaned, stripped };
}

/**
 * Heuristic rewrite: normalize, strip filler, expand acronyms, strip injection, cap length.
 * @param {string} userQuery
 * @param {import('./config.js').config} config
 * @returns {{ rewrittenQuery: string; alternateQueries: string[]; filters: Record<string, unknown>; diagnostics: { intent: string; removed: string[]; stripped: string[] } }}
 */
export function rewriteHeuristic(userQuery, config) {
    const diagnostics = { intent: "", removed: [], stripped: [] };
    let q = normalize(userQuery);
    if (!q) {
        return {
            rewrittenQuery: "",
            alternateQueries: [],
            filters: {},
            diagnostics: { ...diagnostics, intent: "unknown" },
        };
    }

    const intent = classifyIntent(q);
    diagnostics.intent = intent;

    const { cleaned: afterFiller, removed } = stripFiller(q, config.stripPhrases ?? []);
    diagnostics.removed = removed;
    q = afterFiller || q;

    const { cleaned: afterInjection, stripped } = stripInjection(q);
    diagnostics.stripped = stripped;
    q = afterInjection;

    q = expandAcronyms(q, config.acronymMap ?? {});
    const maxLen = config.maxRewriteLength ?? 300;
    if (q.length > maxLen) q = q.slice(0, maxLen).trim();

    const alternateQueries = [];
    if (intent === "comparison" && config.maxAlternates > 0) {
        const vsMatch = q.match(/\b(.+?)\s+(?:vs\.?|versus)\s+(.+)/i);
        if (vsMatch) {
            let a = vsMatch[1].trim();
            let b = vsMatch[2].trim();
            // Drop trailing punctuation from fragments
            a = a.replace(/[?.!]+$/, "");
            b = b.replace(/[?.!]+$/, "");
            if (a && b) {
                alternateQueries.push(a, b);
                if (config.maxAlternates >= 3) alternateQueries.push(`${a} and ${b} comparison`);
            }
        }
    }
    const capped = alternateQueries.slice(0, config.maxAlternates ?? 0);

    const filters = {};
    if (config.allowedFilterKeys?.includes("category") && intent === "troubleshooting") {
        filters.category = "troubleshooting";
    }

    return {
        rewrittenQuery: q,
        alternateQueries: capped,
        filters,
        diagnostics,
    };
}

/**
 * Parse LLM JSON response for rewrite. Expects: { rewrittenQuery, alternateQueries?, filters?, rationale? }
 * @param {string} raw
 * @returns {{ rewrittenQuery: string; alternateQueries: string[]; filters: Record<string, unknown>; rationale?: string } | null}
 */
export function parseLLMRewriteResponse(raw) {
    try {
        const s = raw.trim();
        const jsonMatch = s.match(/\{[\s\S]*\}/);
        if (!jsonMatch) return null;
        const parsed = JSON.parse(jsonMatch[0]);
        const rewrittenQuery = typeof parsed.rewrittenQuery === "string" ? parsed.rewrittenQuery.trim() : "";
        const alternateQueries = Array.isArray(parsed.alternateQueries)
            ? parsed.alternateQueries.filter((x) => typeof x === "string").map((x) => x.trim()).filter(Boolean)
            : [];
        const filters = parsed.filters && typeof parsed.filters === "object" ? parsed.filters : {};
        return {
            rewrittenQuery,
            alternateQueries,
            filters,
            rationale: typeof parsed.rationale === "string" ? parsed.rationale.trim() : undefined,
        };
    } catch {
        return null;
    }
}

/**
 * Full rewrite: optional LLM first, then fallback to heuristic. Validates and caps output.
 * @param {string} userQuery
 * @param {import('./config.js').config} config
 * @param {{ generate?: (prompt: string) => Promise<string>; conversation?: Array<{ role: string; content: string }> }} options - optional LLM and conversation for context
 * @returns {Promise<{ rewrittenQuery: string; alternateQueries: string[]; filters: Record<string, unknown>; diagnostics: { intent: string; removed: string[]; stripped: string[]; rationale?: string; source: 'llm'|'heuristic' } }>}
 */
export async function rewrite(userQuery, config, options = {}) {
    const heuristicResult = rewriteHeuristic(userQuery, config);
    const maxLen = config.maxRewriteLength ?? 300;
    const maxAlt = config.maxAlternates ?? 4;

    if (config.useLLM && typeof options.generate === "function") {
        const prompt = buildRewritePrompt(userQuery, config, options.conversation);
        try {
            const timeoutMs = config.llmTimeoutMs || 0;
            const generate = timeoutMs > 0
                ? () => Promise.race([
                    options.generate(prompt),
                    new Promise((_, rej) => setTimeout(() => rej(new Error("LLM timeout")), timeoutMs)),
                ])
                : () => options.generate(prompt);
            const raw = await generate();
            const parsed = parseLLMRewriteResponse(raw);
            if (parsed && parsed.rewrittenQuery) {
                const rewrittenQuery = parsed.rewrittenQuery.length > maxLen ? parsed.rewrittenQuery.slice(0, maxLen).trim() : parsed.rewrittenQuery;
                const alternateQueries = (parsed.alternateQueries ?? []).slice(0, maxAlt);
                return {
                    rewrittenQuery,
                    alternateQueries,
                    filters: parsed.filters ?? {},
                    diagnostics: {
                        ...heuristicResult.diagnostics,
                        rationale: parsed.rationale,
                        source: "llm",
                    },
                };
            }
        } catch (_) {
            // fall through to heuristic
        }
    }

    return {
        ...heuristicResult,
        diagnostics: {
            ...heuristicResult.diagnostics,
            source: "heuristic",
        },
    };
}

/**
 * Build prompt for LLM rewrite (structured output).
 * @param {string} userQuery
 * @param {import('./config.js').config} config
 * @param {Array<{ role: string; content: string }>} [conversation]
 * @returns {string}
 */
function buildRewritePrompt(userQuery, config, conversation) {
    const filterKeys = (config.allowedFilterKeys || []).join(", ");
    let context = "";
    if (Array.isArray(conversation) && conversation.length > 0) {
        const lastTurns = conversation.slice(-4).map((t) => `${t.role}: ${t.content}`).join("\n");
        context = `Recent conversation:\n${lastTurns}\n\nCurrent user query (resolve "it"/"that" from context if needed):\n`;
    }
    return `You are a query rewriter for a search system. Rewrite the user's query into a single, clear, search-ready query. Do not answer the question; output only a JSON object.

Output format (valid JSON only):
{ "rewrittenQuery": "...", "alternateQueries": ["...", "..."], "filters": {}, "rationale": "optional one-line explanation" }

Rules for rewrittenQuery:
- One clear search query. No filler, no politeness (e.g. no "please", "can you", "thanks").
- Normalize punctuation: at most one question mark or one period at the end (e.g. "???" or "..." become "?" or "."). No repeated "!!" or ",,".
- Use straight quotes. Keep the core question; drop instructions, asides, and off-topic text.
- Max length: ${config.maxRewriteLength ?? 300} characters.
- alternateQueries: 0 to ${config.maxAlternates ?? 4} alternative phrasings (array of strings), also normalized.
- filters: optional object; only use these keys if relevant: ${filterKeys}.
${context ? "- If the user says \"it\" or \"that\", substitute from the recent conversation.\n" : ""}

User query:
${userQuery}

JSON:`;
}
