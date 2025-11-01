/**
 * BUILDING AN LLM WRAPPER CLASS
 *
 * Now that we understand how node-llama-cpp works, let's build
 * a cleaner wrapper class that follows common patterns.
 *
 * This example builds the wrapper step-by-step with explanations.
 */


import {LlamaCpp} from "../../../src/llms/index.js";

console.log('Building an LLM Wrapper Class\n');

console.log(`
WHY BUILD A WRAPPER?
────────────────────

1. Simpler API
   - Hide node-llama-cpp complexity
   - Consistent interface
   - Easy to swap implementations

2. Better Defaults
   - Pre-configured settings
   - Common use case patterns
   - Sensible parameter choices

3. Reusability
   - Use across your RAG system
   - Easy to test and mock
   - Follows standard patterns

4. Future-Proof
   - Easy to add features
   - Can switch backends
   - Maintainable code

Let's build it step by step!
`);

// ============================================================================
// DEMO: Using our wrapper
// ============================================================================

console.log('\n📝 EXAMPLE 1: Basic Usage\n');

const model = await LlamaCpp.initialize({
    modelPath: process.env.MODEL_PATH || './models/llama-3.1-8b-q4_0.gguf',
    maxTokens: 100,
    temperature: 0.7,
});

const response1 = await model.invoke(
    'What are the three primary colors?'
);
console.log('Q: What are the three primary colors?');
console.log('A:', response1);

console.log('\n─────────────────────────────────────────────\n');
console.log('📝 EXAMPLE 2: Streaming Tokens\n');

console.log('Q: Count from 1 to 5.');
console.log('A: ');

for await (const chunk of model.stream('Count from 1 to 5.')) {
    process.stdout.write(chunk);
}

console.log('\n\n─────────────────────────────────────────────\n');
console.log('📝 EXAMPLE 3: Batch Processing\n');

const questions = [
    'What is 2+2?',
    'What color is the sky?',
    'What is the capital of France?',
];

console.log('Processing multiple questions...\n');
const responses = await model.batch(questions);

questions.forEach((q, i) => {
    console.log(`Q${i + 1}: ${q}`);
    console.log(`A${i + 1}: ${responses[i]}\n`);
});

console.log('─────────────────────────────────────────────\n');
console.log('📝 EXAMPLE 4: Custom Options\n');

const response2 = await model.invoke(
    'Write a creative story opening.',
    {
        maxTokens: 150,
        temperature: 0.9,  // More creative
    }
);

console.log('Q: Write a creative story opening.');
console.log('A:', response2);

// Cleanup
await model.dispose();

console.log(`

╔════════════════════════════════════════════════════════════════╗
║                    WRAPPER BENEFITS                            ║
╚════════════════════════════════════════════════════════════════╝

- Clean, simple API
- Easy to understand and use
- Consistent with other frameworks
- Handles resource management
- Supports streaming and batching
- Configurable but with good defaults

WHAT WE BUILT:
──────────────

1. BaseLLM (BaseLLM.js)
   - Abstract base class
   - Defines the interface all LLMs must follow
   - Common methods like batch()

2. LlamaCpp (LlamaCpp.js)
   - Concrete implementation for node-llama-cpp
   - Handles model loading and management
   - Provides invoke(), stream(), batch()

3. Benefits for RAG:
   - Easy to use in retrieval pipelines
   - Can swap for API-based models later
   - Clean separation of concerns

NEXT IN RAG PIPELINE:
- Data loading (how to get documents)
- Text splitting (chunking for embeddings)
- Embeddings (convert text to vectors)
- Vector stores (store and search embeddings)
- Retrieval strategies (find relevant docs)
- Putting it all together (complete RAG system)
`);