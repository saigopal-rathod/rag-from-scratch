/**
 * Multi-query retrieval helpers: parallel retrieval, deduplication, RRF fusion.
 * Framework-agnostic; works with any getEmbedding(text) -> vector and search(namespace, vector, k) -> results.
 */

/**
 * @template T
 * @param {Promise<T>} promise
 * @param {number} ms 0 = no timeout
 * @returns {Promise<T>}
 */
export function withTimeout(promise, ms) {
    if (!ms || ms <= 0) return promise;
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
        ),
    ]);
}

/**
 * @template T
 * @param {() => Promise<T>} fn
 * @param {{ retries?: number; delayMs?: number }} options
 * @returns {Promise<T>}
 */
export async function withRetry(fn, { retries = 1, delayMs = 100 } = {}) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            return await fn();
        } catch (e) {
            lastErr = e;
            if (attempt < retries && delayMs > 0) {
                await new Promise((r) => setTimeout(r, delayMs));
            }
        }
    }
    throw lastErr;
}

/**
 * Deduplicate by document ID; keeps first occurrence.
 * @param {Array<{ metadata: { id: string }; similarity?: number }>} results
 * @returns {Array<{ metadata: { id: string }; similarity?: number }>}
 */
export function deduplicateById(results) {
    return Array.from(new Map(results.map((r) => [r.metadata.id, r])).values());
}

/**
 * Deduplicate by document ID; keep the result with highest similarity per doc.
 * @param {Array<{ metadata: { id: string }; similarity?: number }>} results
 * @returns {Array<{ metadata: { id: string }; similarity?: number }>} Sorted by similarity descending.
 */
export function deduplicateByMaxScore(results) {
    const byId = results.reduce((map, r) => {
        const existing = map.get(r.metadata.id);
        const sim = r.similarity ?? 0;
        if (!existing || (existing.similarity ?? 0) < sim) {
            map.set(r.metadata.id, r);
        }
        return map;
    }, new Map());
    return Array.from(byId.values()).sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0));
}

/**
 * Reciprocal Rank Fusion. Combine multiple ranked lists into one by rank-based scoring.
 * @param {Array<Array<{ metadata: { id: string }; similarity?: number }>>} resultLists
 * @param {number} k RRF constant (e.g. 60)
 * @returns {Array<{ metadata: { id: string }; similarity?: number; rrfScore: number }>} Sorted by rrfScore descending.
 */
export function rrfFuse(resultLists, k = 60) {
    const scores = new Map();
    for (const list of resultLists) {
        list.forEach((doc, rank) => {
            const id = doc.metadata.id;
            const rrfScore = 1 / (k + rank + 1);
            scores.set(id, (scores.get(id) || 0) + rrfScore);
        });
    }
    const flat = resultLists.flat();
    return Array.from(scores.entries())
        .map(([id, rrfScore]) => {
            const doc = flat.find((r) => r.metadata.id === id);
            return { ...doc, rrfScore };
        })
        .sort((a, b) => b.rrfScore - a.rrfScore);
}

/**
 * Run multiple queries in parallel: embed each query, then search. Optional timeout, retry, concurrency cap.
 * @param {string[]} queries
 * @param {{
 *   getEmbedding: (text: string) => Promise<{ vector: number[] | Float32Array }>;
 *   search: (namespace: string, vector: number[], topK: number) => Promise<Array<{ metadata: { id: string }; similarity?: number }>>;
 *   namespace: string;
 *   topK?: number;
 *   concurrency?: number;
 *   timeoutMs?: number;
 *   retries?: number;
 * }} options
 * @returns {Promise<Array<Array<{ metadata: { id: string }; similarity?: number }>>>} One result list per query (order preserved).
 */
export async function retrieveParallel(queries, options) {
    const {
        getEmbedding,
        search,
        namespace,
        topK = 5,
        concurrency = 0,
        timeoutMs = 0,
        retries = 0,
    } = options;

    const runOne = async (query) => {
        const doWork = async () => {
            const emb = await getEmbedding(query);
            const vector = Array.isArray(emb.vector) ? emb.vector : Array.from(emb.vector);
            return search(namespace, vector, topK);
        };
        let p = doWork();
        if (retries > 0) p = withRetry(() => p, { retries, delayMs: 50 });
        if (timeoutMs > 0) p = withTimeout(p, timeoutMs);
        return p;
    };

    if (concurrency > 0) {
        const results = [];
        for (let i = 0; i < queries.length; i += concurrency) {
            const chunk = queries.slice(i, i + concurrency);
            const chunkResults = await Promise.all(chunk.map(runOne));
            results.push(...chunkResults);
        }
        return results;
    }

    return Promise.all(queries.map(runOne));
}
