/**
 * Multi-Query Retrieval for RAG
 *
 * Demonstrates:
 * 1) Query decomposition - Breaking complex queries into sub-queries
 * 2) Parallel retrieval - Executing multiple searches simultaneously
 * 3) Result deduplication and fusion strategies
 * 4) Query expansion - Generating alternative phrasings
 * 5) Perspective-based retrieval - Same query from different angles
 * 6) Handling complex multi-part questions
 * 7) Ranking and merging diverse results
 *
 * Why Multi-Query Retrieval?
 * - Single queries miss relevant information
 * - Complex questions have multiple aspects
 * - Different phrasings retrieve different results
 * - Improves recall and coverage
 *
 * Prerequisites:
 * - npm install embedded-vector-db node-llama-cpp chalk
 * - Place bge-small-en-v1.5.Q8_0.gguf and Qwen3-1.7B-Q8_0.gguf under models/
 */

import { fileURLToPath } from "url";
import path from "path";
import { VectorDB } from "embedded-vector-db";
import { getLlama, LlamaChatSession } from "node-llama-cpp";
import { Document } from "../../../src/index.js";
import { OutputHelper } from "../../../helpers/output-helper.js";
import chalk from "chalk";
import { config } from "./config.js";
import {
    retrieveParallel,
    deduplicateById,
    deduplicateByMaxScore,
    rrfFuse,
} from "./multi-query-retrieval.js";
import { createLogger } from "./logger.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMBEDDING_MODEL_PATH = path.join(__dirname, "..", "..", "..", "models", "bge-small-en-v1.5.Q8_0.gguf");
const LLM_MODEL_PATH = path.join(__dirname, "..", "..", "..", "models", "Qwen3-1.7B-Q8_0.gguf");

const { dim: DIM, maxElements: MAX_ELEMENTS, namespace: NS, rrfK, topKPerQuery } = config;
const logger = createLogger({ logLevel: config.logLevel });

/**
 * Initialize the embedding model
 */
async function initializeEmbeddingModel() {
    try {
        const llama = await getLlama({ logLevel: "error" });
        const model = await llama.loadModel({ modelPath: EMBEDDING_MODEL_PATH });
        return await model.createEmbeddingContext();
    } catch (error) {
        throw new Error(`Failed to initialize embedding model: ${error.message}`);
    }
}

/**
 * Initialize the LLM (Qwen) for query decomposition
 */
async function initializeLLM() {
    try {
        const llama = await getLlama({ logLevel: "error" });
        const model = await llama.loadModel({ modelPath: LLM_MODEL_PATH });
        const context = await model.createContext();
        return new LlamaChatSession({ contextSequence: context.getSequence() });
    } catch (error) {
        throw new Error(`Failed to initialize LLM: ${error.message}`);
    }
}

/** Fallback sub-queries when LLM decomposition fails or is unavailable */
const FALLBACK_SUBQUERIES = [
    "What is web application architecture?",
    "How to make web applications scalable?",
    "What are testing best practices?",
    "How to implement automated testing?",
];

/**
 * Parse LLM output into an array of sub-query strings.
 * Accepts JSON array, JSON object with sub_queries/key, or numbered/bullet lines.
 */
function parseSubQueriesFromLLM(raw) {
    if (!raw || typeof raw !== "string") return null;
    const s = raw.trim();
    const jsonMatch = s.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
    if (jsonMatch) {
        try {
            const parsed = JSON.parse(jsonMatch[0]);
            const list = Array.isArray(parsed)
                ? parsed
                : parsed.sub_queries ?? parsed.subQueries ?? parsed.queries ?? null;
            if (Array.isArray(list) && list.length > 0) {
                const queries = list.filter((q) => typeof q === "string").map((q) => q.trim()).filter(Boolean);
                if (queries.length > 0) return queries;
            }
        } catch (_) {}
    }
    const lines = s.split(/\n/).map((line) => line.replace(/^\s*[\d\-*.]+\s*/, "").trim()).filter(Boolean);
    if (lines.length > 0) return lines;
    return null;
}

/**
 * Decompose a complex query into 3-5 focused sub-queries using the LLM (Qwen).
 * On failure or parse error, returns fallback list.
 */
async function decomposeQueryWithLLM(complexQuery, chatSession) {
    const prompt = `You are a search query assistant. Given a complex question, break it into 3–5 focused sub-questions that together cover all aspects of the original. Output only a JSON array of strings, one sub-question per element. Do not include explanations or extra text.

Complex question: "${complexQuery}"

JSON array:`;
    try {
        const response = await chatSession.prompt(prompt, { maxTokens: 400 });
        const parsed = parseSubQueriesFromLLM(response);
        if (parsed && parsed.length >= 2) {
            return parsed.slice(0, config.maxSubQueries);
        }
    } catch (_) {}
    return FALLBACK_SUBQUERIES;
}

/**
 * Create a comprehensive knowledge base about software development
 */
function createKnowledgeBase() {
    return [
        // Frontend Development
        new Document("React is a JavaScript library for building user interfaces using components. It uses a virtual DOM for efficient updates and supports hooks for state management.", {
            id: "doc_001",
            category: "frontend",
            topic: "react",
            subtopic: "basics"
        }),
        new Document("React hooks like useState and useEffect allow functional components to manage state and side effects. They replaced class components for most use cases.", {
            id: "doc_002",
            category: "frontend",
            topic: "react",
            subtopic: "hooks"
        }),
        new Document("React performance optimization includes memoization with useMemo and useCallback, code splitting with React.lazy, and virtual scrolling for large lists.", {
            id: "doc_003",
            category: "frontend",
            topic: "react",
            subtopic: "performance"
        }),
        new Document("Vue.js provides a reactive data system with a template-based syntax. The Composition API offers flexible component logic organization similar to React hooks.", {
            id: "doc_004",
            category: "frontend",
            topic: "vue",
            subtopic: "basics"
        }),
        new Document("Next.js is a React framework offering server-side rendering, static site generation, API routes, and file-based routing. It improves SEO and initial page load.", {
            id: "doc_005",
            category: "frontend",
            topic: "nextjs",
            subtopic: "features"
        }),

        // Backend Development
        new Document("Node.js enables JavaScript on the server using the V8 engine. It excels at I/O-bound tasks with its event-driven, non-blocking architecture.", {
            id: "doc_006",
            category: "backend",
            topic: "nodejs",
            subtopic: "basics"
        }),
        new Document("Express.js is a minimal Node.js web framework providing routing, middleware support, and HTTP utility methods. It's the foundation for many Node applications.", {
            id: "doc_007",
            category: "backend",
            topic: "express",
            subtopic: "basics"
        }),
        new Document("GraphQL APIs allow clients to request exactly the data they need. Schema definition, resolvers, and strong typing prevent over-fetching and under-fetching.", {
            id: "doc_008",
            category: "backend",
            topic: "graphql",
            subtopic: "api-design"
        }),
        new Document("REST API best practices include versioning, proper HTTP methods, pagination, rate limiting, and comprehensive error responses with status codes.", {
            id: "doc_009",
            category: "backend",
            topic: "rest",
            subtopic: "best-practices"
        }),
        new Document("Microservices architecture splits applications into independent services. Each service handles specific business capabilities and can be deployed separately.", {
            id: "doc_010",
            category: "backend",
            topic: "microservices",
            subtopic: "architecture"
        }),

        // Database
        new Document("PostgreSQL offers advanced features like JSONB support, full-text search, window functions, and ACID compliance. It handles complex queries efficiently.", {
            id: "doc_011",
            category: "database",
            topic: "postgresql",
            subtopic: "features"
        }),
        new Document("MongoDB is a document database storing data in flexible JSON-like documents. It supports horizontal scaling through sharding and has a rich query language.", {
            id: "doc_012",
            category: "database",
            topic: "mongodb",
            subtopic: "basics"
        }),
        new Document("Database indexing dramatically improves query performance. B-tree indexes work for range queries, while hash indexes excel at equality comparisons.", {
            id: "doc_013",
            category: "database",
            topic: "optimization",
            subtopic: "indexing"
        }),
        new Document("Database normalization reduces redundancy by organizing data into related tables. However, denormalization can improve read performance for specific use cases.", {
            id: "doc_014",
            category: "database",
            topic: "design",
            subtopic: "normalization"
        }),
        new Document("Redis is an in-memory data store used for caching, session management, and real-time analytics. It supports various data structures like strings, lists, sets, and hashes.", {
            id: "doc_015",
            category: "database",
            topic: "redis",
            subtopic: "caching"
        }),

        // Testing
        new Document("Unit testing verifies individual functions in isolation using frameworks like Jest or Mocha. Tests should be fast, independent, and cover edge cases.", {
            id: "doc_016",
            category: "testing",
            topic: "unit-testing",
            subtopic: "basics"
        }),
        new Document("Integration testing checks how components work together. It tests database interactions, API calls, and service integrations with realistic data.", {
            id: "doc_017",
            category: "testing",
            topic: "integration-testing",
            subtopic: "basics"
        }),
        new Document("Test-driven development (TDD) writes tests before implementation. The red-green-refactor cycle ensures code meets requirements and remains maintainable.", {
            id: "doc_018",
            category: "testing",
            topic: "tdd",
            subtopic: "methodology"
        }),
        new Document("End-to-end testing simulates real user workflows using tools like Cypress or Playwright. It validates the entire application stack from UI to database.", {
            id: "doc_019",
            category: "testing",
            topic: "e2e-testing",
            subtopic: "automation"
        }),
        new Document("Code coverage measures which lines are executed during tests. While useful, 100% coverage doesn't guarantee quality - focus on meaningful test cases.", {
            id: "doc_020",
            category: "testing",
            topic: "coverage",
            subtopic: "metrics"
        }),

        // DevOps
        new Document("Docker containers package applications with dependencies for consistent deployment. Images are built from Dockerfiles and run isolated from the host system.", {
            id: "doc_021",
            category: "devops",
            topic: "docker",
            subtopic: "basics"
        }),
        new Document("Kubernetes orchestrates containerized applications across clusters. It handles scaling, load balancing, rolling updates, and self-healing automatically.", {
            id: "doc_022",
            category: "devops",
            topic: "kubernetes",
            subtopic: "orchestration"
        }),
        new Document("CI/CD pipelines automate testing and deployment. Continuous integration catches bugs early, while continuous deployment ensures rapid, reliable releases.", {
            id: "doc_023",
            category: "devops",
            topic: "cicd",
            subtopic: "automation"
        }),
        new Document("Infrastructure as Code (IaC) manages infrastructure through version-controlled configuration files. Tools like Terraform enable reproducible, auditable deployments.", {
            id: "doc_024",
            category: "devops",
            topic: "iac",
            subtopic: "terraform"
        }),
        new Document("Monitoring and logging are crucial for production systems. Metrics track system health, logs provide detailed events, and alerts notify teams of issues.", {
            id: "doc_025",
            category: "devops",
            topic: "observability",
            subtopic: "monitoring"
        })
    ];
}

/**
 * Add documents to vector store
 */
async function addDocumentsToStore(vectorStore, embeddingContext, documents) {
    try {
        for (const doc of documents) {
            const embedding = await embeddingContext.getEmbeddingFor(doc.pageContent);
            const metadata = {
                content: doc.pageContent,
                ...doc.metadata,
            };
            await vectorStore.insert(
                NS,
                doc.metadata.id,
                Array.from(embedding.vector),
                metadata
            );
        }
    } catch (error) {
        throw new Error(`Failed to add documents to store: ${error.message}`);
    }
}

/**
 * Example 1: Query Decomposition
 * Break complex queries into simpler sub-queries (LLM with Qwen via node-llama-cpp)
 */
async function example1(embeddingContext, vectorStore, chatSession) {
    const complexQuery = "What are the best practices for designing scalable architectures and implementing effective testing strategies?";
    console.log(`${chalk.bold("Complex Query:")} "${complexQuery}"\n`);

    const llmSubQueries = await decomposeQueryWithLLM(complexQuery, chatSession);
    // Ensure we have a microservices-focused sub-query for this example
    const subQueries = [];
    const maxSubs = config.maxSubQueries ?? 7;
    for (const q of llmSubQueries) {
        if (subQueries.length >= maxSubs - 1) break;
        subQueries.push(q);
    }

    console.log(chalk.bold("Decomposed Sub-Queries:"));
    subQueries.forEach((sq, idx) => {
        console.log(`  ${idx + 1}. "${sq}"`);
    });
    console.log();

    // Single query retrieval
    console.log(chalk.bold("Approach 1: Single Query"));
    const singleResults = await logger.timeAsync("single-query", async () => {
        const emb = await embeddingContext.getEmbeddingFor(complexQuery);
        return vectorStore.search(NS, Array.from(emb.vector), 3);
    });

    console.log("Results:");
    singleResults.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
        console.log(`     ${chalk.dim(doc.metadata.content.substring(0, 60) + '...')}`);
    });

    // Multi-query retrieval (parallel via helper)
    console.log(`\n${chalk.bold("Approach 2: Multi-Query (Decomposed)")}`);
    const resultLists = await logger.timeAsync("multi-query", () =>
        retrieveParallel(subQueries, {
            getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
            search: (ns, vector, k) => vectorStore.search(ns, vector, k),
            namespace: NS,
            topK: 4,
            retries: config.retries,
            timeoutMs: config.timeoutMs || undefined,
        })
    );
    const fusedResults = rrfFuse(resultLists, rrfK);

    console.log(`Results (${fusedResults.length} unique documents, RRF-ranked):`);
    fusedResults.slice(0, 8).forEach((doc, idx) => {
        const score = doc.rrfScore ?? doc.similarity ?? 0;
        console.log(`  ${idx + 1}. [${chalk.green(score.toFixed(4))}] ${doc.metadata.topic}`);
        console.log(`     ${chalk.dim(doc.metadata.content.substring(0, 60) + '...')}`);
    });

    console.log(`\n${chalk.bold("Key Insight:")}`);
    console.log("Multi-query retrieval found more diverse, relevant results:");
    console.log(`• Single query: ${singleResults.length} results (may miss aspects)`);
    console.log(`• Multi-query: ${fusedResults.length} unique results (RRF-ranked, comprehensive coverage)`);
    console.log("• Documents that support multiple aspects (e.g. architecture and scalability) rank higher.\n");
}

/**
 * Example 2: Parallel Retrieval with Query Expansion
 * Generate multiple phrasings and search simultaneously
 */
async function example2(embeddingContext, vectorStore) {
    const originalQuery = "How to optimize database queries?";
    console.log(`${chalk.bold("Original Query:")} "${originalQuery}"\n`);

    // Generate alternative phrasings
    const expandedQueries = [
        originalQuery,
        "What are database performance optimization techniques?",
        "Ways to improve database query speed",
        "Database indexing and query optimization strategies"
    ];

    console.log(chalk.bold("Expanded Queries:"));
    expandedQueries.forEach((q, idx) => {
        console.log(`  ${idx + 1}. "${q}"`);
    });
    console.log();

    // Sequential execution (slow)
    console.log(chalk.bold("Sequential Execution:"));
    const startSeq = Date.now();
    const seqResults = [];
    for (const query of expandedQueries) {
        const emb = await embeddingContext.getEmbeddingFor(query);
        const results = await vectorStore.search(NS, Array.from(emb.vector), 3);
        seqResults.push(...results);
    }
    const seqTimeMs = Date.now() - startSeq;
    console.log(`Time: ${seqTimeMs}ms\n`);

    // Parallel execution (fast) via helper
    console.log(chalk.bold("Parallel Execution:"));
    const startPar = Date.now();
    const parResultLists = await retrieveParallel(expandedQueries, {
        getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
        search: (ns, vector, k) => vectorStore.search(ns, vector, k),
        namespace: NS,
        topK: 3,
        retries: config.retries,
        timeoutMs: config.timeoutMs || undefined,
    });
    const parTimeMs = Date.now() - startPar;
    console.log(`Time: ${parTimeMs}ms`);
    console.log(`${chalk.green("Speedup:")} ${(seqTimeMs / parTimeMs).toFixed(1)}x faster\n`);

    const parResults = parResultLists.flat();
    const uniqueResults = deduplicateById(parResults);
    logger.metrics({ queryCount: expandedQueries.length, totalResults: parResults.length, uniqueAfterDedup: uniqueResults.length, durationMs: parTimeMs });

    console.log(chalk.bold(`Unique Results (${uniqueResults.length} documents):`));
    uniqueResults.slice(0, 4).forEach((doc, idx) => {
        console.log(`  ${idx + 1}. ${doc.metadata.topic} - ${doc.metadata.subtopic}`);
        console.log(`     ${chalk.dim(doc.metadata.content.substring(0, 70) + '...')}`);
    });

    console.log(`\n${chalk.bold("Key Insight:")}`);
    console.log("Query expansion + parallel execution gives:");
    console.log("• Better coverage through different phrasings");
    console.log("• Faster execution with Promise.all()");
    console.log("• More robust to query phrasing variations\n");
}

/**
 * Example 3: Reciprocal Rank Fusion (RRF)
 * Combine multiple result lists using rank-based scoring
 */
async function example3(embeddingContext, vectorStore) {
    const queries = [
        "React performance optimization",
        "How to make React apps faster?",
        "React rendering optimization techniques"
    ];

    console.log(chalk.bold("Multiple Query Variations:"));
    queries.forEach((q, idx) => {
        console.log(`  ${idx + 1}. "${q}"`);
    });
    console.log();

    // Execute all queries in parallel
    const allResults = await retrieveParallel(queries, {
        getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
        search: (ns, vector, k) => vectorStore.search(ns, vector, k),
        namespace: NS,
        topK: topKPerQuery,
        retries: config.retries,
        timeoutMs: config.timeoutMs || undefined,
    });

    // Show individual result lists
    console.log(chalk.bold("Individual Query Results:\n"));
    allResults.forEach((results, qIdx) => {
        console.log(`Query ${qIdx + 1}:`);
        results.slice(0, 3).forEach((doc, idx) => {
            console.log(`  ${idx + 1}. [rank ${idx + 1}] ${doc.metadata.id}: ${doc.metadata.topic}`);
        });
        console.log();
    });

    // Apply Reciprocal Rank Fusion via helper
    const fusedResults = rrfFuse(allResults, rrfK);

    console.log(chalk.bold("Fused Results (RRF):"));
    fusedResults.slice(0, 5).forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.rrfScore.toFixed(4))}] ${doc.metadata.id}: ${doc.metadata.topic}`);
        console.log(`     ${chalk.dim(doc.metadata.content.substring(0, 60) + '...')}`);
    });

    console.log(`\n${chalk.bold("Key Insight:")}`);
    console.log("RRF Formula: score = Σ 1/(k + rank)");
    console.log("• Documents appearing in multiple queries rank higher");
    console.log("• Rank-based scoring is robust to score scale differences");
    console.log(`• k=${rrfK} (configurable); higher k gives more weight to top ranks\n`);
}

/**
 * Example 4: Perspective-Based Retrieval
 * Same topic from different viewpoints
 */
async function example4(embeddingContext, vectorStore) {
    const topic = "microservices architecture";
    console.log(`${chalk.bold("Topic:")} ${topic}\n`);

    // Different perspectives on the same topic
    const perspectives = [
        { query: "What are the benefits of microservices?", label: "Benefits" },
        { query: "What are the challenges of microservices?", label: "Challenges" },
        { query: "How to implement microservices?", label: "Implementation" },
        { query: "When to use microservices vs monolith?", label: "Comparison" }
    ];

    console.log(chalk.bold("Different Perspectives:"));
    perspectives.forEach((p, idx) => {
        console.log(`  ${idx + 1}. ${chalk.cyan(p.label)}: "${p.query}"`);
    });
    console.log();

    const perspectiveQueries = perspectives.map((p) => p.query);
    const perspectiveResultLists = await retrieveParallel(perspectiveQueries, {
        getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
        search: (ns, vector, k) => vectorStore.search(ns, vector, k),
        namespace: NS,
        topK: 2,
        retries: config.retries,
        timeoutMs: config.timeoutMs || undefined,
    });
    const perspectiveResults = perspectives.map((p, i) => ({
        perspective: p.label,
        results: perspectiveResultLists[i] ?? [],
    }));

    console.log(chalk.bold("Results by Perspective:\n"));
    perspectiveResults.forEach((pr) => {
        console.log(chalk.bold(`${pr.perspective}:`));
        pr.results.forEach((doc, idx) => {
            console.log(`  ${idx + 1}. ${doc.metadata.topic} (${doc.metadata.category})`);
            console.log(`     ${chalk.dim(doc.metadata.content.substring(0, 70) + '...')}`);
        });
        console.log();
    });

    // Combine all perspectives
    const allDocs = perspectiveResults.flatMap((pr) => pr.results);
    const uniqueDocs = deduplicateById(allDocs);

    console.log(chalk.bold(`Combined Coverage: ${uniqueDocs.length} unique documents`));

    console.log(`\n${chalk.bold("Key Insight:")}`);
    console.log("Perspective-based retrieval provides:");
    console.log("• Comprehensive coverage of a topic");
    console.log("• Multiple viewpoints (pros, cons, how-to)");
    console.log("• Balanced information for decision-making");
    console.log("• Better answers to nuanced questions\n");
}

/**
 * Example 5: Multi-Part Question Handling
 * Break down questions with multiple distinct parts
 */
async function example5(embeddingContext, vectorStore) {
    const multiPartQuery = "What's the difference between unit testing and integration testing, and when should I use each?";
    console.log(`${chalk.bold("Multi-Part Query:")}`);
    console.log(`"${multiPartQuery}"\n`);

    // Identify distinct parts
    const parts = [
        { query: "What is unit testing?", label: "Unit Testing Definition" },
        { query: "What is integration testing?", label: "Integration Testing Definition" },
        { query: "When to use unit testing?", label: "Unit Testing Use Cases" },
        { query: "When to use integration testing?", label: "Integration Testing Use Cases" },
        { query: "Difference between unit and integration testing", label: "Comparison" }
    ];

    console.log(chalk.bold("Identified Parts:"));
    parts.forEach((p, idx) => {
        console.log(`  ${idx + 1}. ${chalk.cyan(p.label)}`);
        console.log(`     ${chalk.dim(p.query)}`);
    });
    console.log();

    // Retrieve for each part in parallel
    const partQueries = parts.map((p) => p.query);
    const partResultLists = await retrieveParallel(partQueries, {
        getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
        search: (ns, vector, k) => vectorStore.search(ns, vector, k),
        namespace: NS,
        topK: 2,
        retries: config.retries,
        timeoutMs: config.timeoutMs || undefined,
    });
    const partResults = parts.map((p, i) => ({
        part: p.label,
        results: partResultLists[i] ?? [],
    }));

    console.log(chalk.bold("Results by Question Part:\n"));
    partResults.forEach((pr) => {
        console.log(chalk.bold(`${pr.part}:`));
        pr.results.forEach((doc, idx) => {
            console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
            console.log(`     ${chalk.dim(doc.metadata.content.substring(0, 80) + '...')}`);
        });
        console.log();
    });

    console.log(`${chalk.bold("Key Insight:")}`);
    console.log("Multi-part decomposition ensures:");
    console.log("• Each aspect of the question is addressed");
    console.log("• No part of the question is ignored");
    console.log("• Comprehensive answers to complex queries");
    console.log("• Better handling of 'and' / 'or' questions\n");
}

/**
 * Example 6: Weighted Query Fusion
 * Combine results with different importance weights
 */
async function example6(embeddingContext, vectorStore) {
    const mainQuery = "React state management";
    const relatedQueries = [
        { query: "React hooks for state", weight: 0.8 },
        { query: "Redux state management", weight: 0.5 },
        { query: "Context API React", weight: 0.6 }
    ];

    console.log(`${chalk.bold("Main Query:")} "${mainQuery}"`);
    console.log(`\n${chalk.bold("Related Queries (with weights):")}`);
    relatedQueries.forEach((rq, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(`w=${rq.weight}`)}] "${rq.query}"`);
    });
    console.log();

    // Retrieve for main query
    const mainEmbedding = await embeddingContext.getEmbeddingFor(mainQuery);
    const mainResults = await vectorStore.search(NS, Array.from(mainEmbedding.vector), 5);

    // Retrieve for related queries in parallel
    const relatedQueriesOnly = relatedQueries.map((rq) => rq.query);
    const relatedResultLists = await retrieveParallel(relatedQueriesOnly, {
        getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
        search: (ns, vector, k) => vectorStore.search(ns, vector, k),
        namespace: NS,
        topK: 3,
        retries: config.retries,
        timeoutMs: config.timeoutMs || undefined,
    });
    const relatedResults = relatedQueries.map((rq, i) => ({
        results: relatedResultLists[i] ?? [],
        weight: rq.weight,
    }));

    // Calculate weighted scores
    const weightedScores = new Map();

    // Main query results (weight = 1.0)
    mainResults.forEach((doc) => {
        weightedScores.set(doc.metadata.id, {
            doc: doc,
            score: doc.similarity * 1.0
        });
    });

    // Related query results (with their weights)
    relatedResults.forEach(({ results, weight }) => {
        results.forEach((doc) => {
            const existing = weightedScores.get(doc.metadata.id);
            if (existing) {
                existing.score += doc.similarity * weight;
            } else {
                weightedScores.set(doc.metadata.id, {
                    doc: doc,
                    score: doc.similarity * weight
                });
            }
        });
    });

    // Sort by weighted score
    const fusedResults = Array.from(weightedScores.values())
        .sort((a, b) => b.score - a.score);

    console.log(chalk.bold("Weighted Fusion Results:"));
    fusedResults.slice(0, 5).forEach((item, idx) => {
        console.log(`  ${idx + 1}. [${chalk.green(item.score.toFixed(4))}] ${item.doc.metadata.topic}`);
        console.log(`     ${chalk.dim(item.doc.metadata.content.substring(0, 70) + '...')}`);
    });

    console.log(`\n${chalk.bold("Key Insight:")}`);
    console.log("Weighted fusion allows:");
    console.log("• Prioritizing the main query over related ones");
    console.log("• Incorporating context without overwhelming main results");
    console.log("• Fine-tuned control over result ranking");
    console.log("• Balancing precision (main) and recall (related)\n");
}

/**
 * Example 7: Adaptive Multi-Query Strategy
 * Choose strategy based on query complexity
 */
async function example7(embeddingContext, vectorStore) {
    /**
     * Analyze query and recommend strategy
     */
    function analyzeQueryComplexity(query) {
        const wordCount = query.split(/\s+/).length;
        const hasMultipleTopics = /\band\b|\bor\b|\bvs\b|\bcompare\b/i.test(query);
        const isQuestion = /\?$/.test(query);
        const hasComparison = /difference|compare|versus|vs/i.test(query);

        let complexity, strategy;

        if (hasMultipleTopics && wordCount > 15) {
            complexity = "Very Complex";
            strategy = "Multi-Part Decomposition";
        } else if (hasComparison) {
            complexity = "Complex";
            strategy = "Perspective-Based Retrieval";
        } else if (wordCount > 10 && isQuestion) {
            complexity = "Moderate";
            strategy = "Query Expansion";
        } else {
            complexity = "Simple";
            strategy = "Single Query (no decomposition needed)";
        }

        return { complexity, strategy, features: { wordCount, hasMultipleTopics, hasComparison, isQuestion } };
    }

    const testQueries = [
        "What is React?",
        "How do I optimize React performance?",
        "What's the difference between REST and GraphQL?",
        "How do I build a scalable web app with microservices, implement CI/CD, and ensure good test coverage?"
    ];

    console.log(chalk.bold("Query Complexity Analysis:\n"));

    for (const query of testQueries) {
        const analysis = analyzeQueryComplexity(query);

        console.log(`${chalk.bold("Query:")} "${query}"`);
        console.log(`${chalk.cyan("Complexity:")} ${analysis.complexity}`);
        console.log(`${chalk.cyan("Recommended Strategy:")} ${analysis.strategy}`);
        console.log(`${chalk.dim("Features:")} ${JSON.stringify(analysis.features)}`);

        // Execute with recommended strategy
        let results;
        const embedding = await embeddingContext.getEmbeddingFor(query);

        if (analysis.strategy === "Single Query (no decomposition needed)") {
            results = await vectorStore.search(NS, Array.from(embedding.vector), 3);
        } else if (analysis.strategy === "Query Expansion") {
            const queries = [query, `How to ${query.replace(/^How (do|to) I /i, "")}`].slice(0, config.maxSubQueries);
            const resultLists = await retrieveParallel(queries, {
                getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
                search: (ns, vector, k) => vectorStore.search(ns, vector, k),
                namespace: NS,
                topK: 2,
                retries: config.retries,
                timeoutMs: config.timeoutMs || undefined,
            });
            results = deduplicateById(resultLists.flat());
        } else {
            // For complex queries, use more retrieval
            results = await vectorStore.search(NS, Array.from(embedding.vector), 5);
        }

        console.log(`${chalk.green("Results:")} ${results.length} documents retrieved`);
        results.slice(0, 2).forEach((doc, idx) => {
            console.log(`  ${idx + 1}. ${doc.metadata.topic} (${doc.metadata.category})`);
        });
        console.log();
    }

    console.log(`${chalk.bold("Key Insight:")}`);
    console.log("Adaptive strategy selection:");
    console.log("• Simple queries - Single retrieval (fast)");
    console.log("• Moderate queries - Query expansion (better coverage)");
    console.log("• Complex queries - Perspective-based or decomposition (comprehensive)");
    console.log("• Very complex - Multi-part decomposition (thorough)");
    console.log("Balances performance and quality based on need\n");
}

/**
 * Example 8: Deduplication Strategies
 * Different approaches to remove duplicate results
 */
async function example8(embeddingContext, vectorStore) {
    const queries = [
        "React state management hooks",
        "useState and useEffect in React",
        "React hooks for managing state"
    ];

    // Retrieve all in parallel
    const resultLists = await retrieveParallel(queries, {
        getEmbedding: (text) => embeddingContext.getEmbeddingFor(text),
        search: (ns, vector, k) => vectorStore.search(ns, vector, k),
        namespace: NS,
        topK: 4,
        retries: config.retries,
        timeoutMs: config.timeoutMs || undefined,
    });
    const allResults = resultLists.flat();

    console.log(`${chalk.bold("Total Results:")} ${allResults.length} (with duplicates)\n`);

    // Strategy 1: By Document ID
    console.log(chalk.bold("Strategy 1: Deduplicate by Document ID"));
    const byId = deduplicateById(allResults);
    console.log(`Unique documents: ${byId.length}`);
    byId.slice(0, 3).forEach((doc, idx) => {
        console.log(`  ${idx + 1}. ${doc.metadata.id}: ${doc.metadata.topic}`);
    });

    // Strategy 2: By Topic + Category
    console.log(`\n${chalk.bold("Strategy 2: Deduplicate by Topic + Category")}`);
    const byTopicCategory = Array.from(
        new Map(allResults.map(r => [
            `${r.metadata.topic}-${r.metadata.category}`, r
        ])).values()
    );
    console.log(`Unique topic-category pairs: ${byTopicCategory.length}`);
    byTopicCategory.slice(0, 3).forEach((doc, idx) => {
        console.log(`  ${idx + 1}. ${doc.metadata.topic} (${doc.metadata.category})`);
    });

    // Strategy 3: Keep highest score per document
    console.log(`\n${chalk.bold("Strategy 3: Keep Highest Score per Document")}`);
    const byMaxScore = deduplicateByMaxScore(allResults);

    console.log(`Unique documents (best score): ${byMaxScore.length}`);
    byMaxScore.slice(0, 3).forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
    });

    console.log(`\n${chalk.bold("Key Insight:")}`);
    console.log("Choose deduplication strategy based on needs:");
    console.log("• By ID: Simple, fast, preserves first occurrence");
    console.log("• By topic: Groups similar content together");
    console.log("• By max score: Keeps best match for each document");
    console.log("Max score strategy often best for ranking quality\n");
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

async function runAllExamples() {
    console.clear();
    console.log("\n" + "=".repeat(80));
    console.log(chalk.bold("RAG from Scratch - Multi-Query Retrieval"));
    console.log("=".repeat(80) + "\n");

    console.log(chalk.dim("What you'll learn:"));
    console.log(chalk.dim("• Query decomposition for complex questions"));
    console.log(chalk.dim("• Parallel retrieval for speed"));
    console.log(chalk.dim("• Result fusion strategies (RRF, weighted)"));
    console.log(chalk.dim("• Perspective-based and multi-part retrieval"));
    console.log(chalk.dim("• Adaptive strategy selection"));
    console.log(chalk.dim("• Deduplication techniques\n"));

    try {
        const embeddingContext = await OutputHelper.withSpinner(
            "Loading embedding model...",
            () => initializeEmbeddingModel()
        );

        const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
        const documents = createKnowledgeBase();

        await OutputHelper.withSpinner(
            "Building knowledge base...",
            () => addDocumentsToStore(vectorStore, embeddingContext, documents)
        );

        const chatSession = await OutputHelper.withSpinner(
            "Loading LLM (Qwen) for query decomposition...",
            () => initializeLLM()
        );

        await OutputHelper.runExample("Example 1: Query Decomposition", () => example1(embeddingContext, vectorStore, chatSession));
        await OutputHelper.runExample("Example 2: Query Expansion & Parallel Retrieval", () => example2(embeddingContext, vectorStore));
        await OutputHelper.runExample("Example 3: Reciprocal Rank Fusion", () => example3(embeddingContext, vectorStore));
        await OutputHelper.runExample("Example 4: Perspective-Based Retrieval", () => example4(embeddingContext, vectorStore));
        await OutputHelper.runExample("Example 5: Multi-Part Questions", () => example5(embeddingContext, vectorStore));
        await OutputHelper.runExample("Example 6: Weighted Query Fusion", () => example6(embeddingContext, vectorStore));
        await OutputHelper.runExample("Example 7: Adaptive Strategy", () => example7(embeddingContext, vectorStore));
        await OutputHelper.runExample("Example 8: Deduplication", () => example8(embeddingContext, vectorStore));

        console.log(chalk.bold.green("\nAll examples completed successfully!\n"));

        console.log(chalk.bold("Key Takeaways:"));
        console.log("• Multi-query retrieval improves recall and coverage");
        console.log("• Query decomposition handles complex questions");
        console.log("• Parallel execution speeds up multiple queries");
        console.log("• RRF combines rankings robustly");
        console.log("• Weighted fusion balances different query types");
        console.log("• Adaptive strategies optimize for query complexity");
        console.log("• Proper deduplication maintains result quality\n");

        console.log(chalk.bold("When to Use Multi-Query:"));
        console.log("┌─────────────────────────────┬──────────────────────┐");
        console.log("│ Scenario                    │ Strategy             │");
        console.log("├─────────────────────────────┼──────────────────────┤");
        console.log("│ Complex, multi-aspect query │ Query Decomposition  │");
        console.log("│ Want better coverage        │ Query Expansion      │");
        console.log("│ Comparison questions        │ Perspective-Based    │");
        console.log("│ Multiple distinct parts     │ Multi-Part Handling  │");
        console.log("│ Uncertain phrasing          │ Query Variations     │");
        console.log("└─────────────────────────────┴──────────────────────┘\n");

        console.log(chalk.bold("Best Practices:"));
        console.log("- Use parallel execution (Promise.all) for speed");
        console.log("- Deduplicate by max score for ranking quality");
        console.log("- Apply RRF when score scales differ");
        console.log("- Weight main query higher than expansions");
        console.log("- Limit sub-queries to 3-5 for efficiency");
        console.log("- Adapt strategy to query complexity");
        console.log("- Monitor latency vs quality tradeoffs\n");

        console.log(chalk.bold("Result Fusion Methods:"));
        console.log("• Simple Union: Fast, but no ranking");
        console.log("• Score Averaging: Simple, but sensitive to scales");
        console.log("• Reciprocal Rank Fusion: Robust, rank-based");
        console.log("• Weighted Fusion: Flexible, controllable");
        console.log("RRF recommended for most cases\n");

    } catch (error) {
        console.error(chalk.red("\nError:"), error?.message ?? error);
        console.error(chalk.dim("\nMake sure you have:"));
        console.error(chalk.dim("1. Installed: npm install embedded-vector-db node-llama-cpp chalk"));
        console.error(chalk.dim("2. Model files in models/: bge-small-en-v1.5.Q8_0.gguf, Qwen3-1.7B-Q8_0.gguf"));
        console.error(chalk.dim("3. Completed previous examples\n"));
    }

    process.exit(0);
}

runAllExamples();