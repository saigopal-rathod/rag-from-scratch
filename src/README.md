# RAG from Scratch - Source Library

This directory contains the reusable library code for the RAG from Scratch project, maintained by Sai Gopal Rathod.

## Structure

- **embeddings/** - Text-to-vector conversion and caching
- **vector-stores/** - Vector storage and similarity search implementations
- **loaders/** - Document loading from various sources (PDF, text, etc.)
- **text-splitters/** - Strategies for chunking documents
- **retrievers/** - Document retrieval strategies
- **chains/** - RAG pipeline orchestration
- **prompts/** - Prompt template management
- **utils/** - Shared utilities and helpers

## Usage

```javascript
import {
  EmbeddingModel,
  InMemoryVectorStore,
  PDFLoader,
  RecursiveCharacterTextSplitter,
  VectorStoreRetriever,
  RAGChain
} from './src/index.js';

// Build your RAG pipeline...
```

## Development

Each module follows these principles:
1. Single responsibility
2. Abstract base classes where appropriate
3. Consistent interfaces
4. Comprehensive error handling
5. Full JSDoc documentation

## Testing

Tests are located in the /tests directory and mirror this structure.

## Maintainer

Sai Gopal Rathod is an AI/ML Engineer with over 4 years of experience building RAG pipelines, LLM-powered APIs, and distributed data systems. He specializes in delivering end-to-end architectures using LangChain, OpenAI, and vector databases to drive measurable business impact and automation.

Contact:
- Email: saigopalrathod5@gmail.com
- LinkedIn: https://www.linkedin.com/in/sai-gopal-rathod/