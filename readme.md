# Multi-modal RAG for scientific literature QA

Purpose of this project is to develop a pipeline for large scale scientific paper processing and QA. The intent is to incoroporate this into an autonomous agentic research agent. 

# Progress
| Stage                                            | Status        |
|--------------------------------------------------|---------------|
| PDF ingestion (MinerU)                           | Complete      |
| Hybrid chunking + contextual prefixing           | Complete      |
| Graph construction (entity & relation extraction)| Complete      |
| Multimodal block handling (figures, equations)   | Complete      |
| Retrieval pipeline                               | Complete      |
| Generation (vLLM + Gemma4)                       | Complete      |
| ColPali visual retrieval                         | Planned       |
| RAGAS evaluation harness                         | In Progress   |


## Architecture 

## Document ingestion
- Mineru processing of PDF files generates blocks of information. Docling was tested but failed to extract clean blocks of text needed for scientific document analysis but woudl be considered if this project were to be applyed to financial document / report analysis.

## 2. Chunking
- Blocks merged into chunks using a hybrid chunker combining token limits and textual flags
- Contextual prefix generation - appending short sentence describing context of chunk in parent document
- TODO
    - Handle multi-modal block types (figures, graphs, equations etc). Figures and graphs will be handled separately
    - equations and all other block types will be incorporated directly into chunk.

## 3. Graph construction 
- LLM entity and relationship extraction from chunks
- Chunk summaries, entity descriptions embedded and added to graph with relationship edges
Graph preferenced over standard naive RAG due to need for complex multi step reasoning required for extracting and condensing information from various locations in a document but also between documents (reveal contradictions, propose further research directions etc)

## 4. Retrieval
- TODO
- Query passed through router to classify query type, dictates specific pipeline
- Queries requiring figure/graph interpretation pass whole page to Colpali embedder
- Initial retrieval based on cosine similarity of query with query decomp. 
- Semantic graph traversal to detect relevant chunks that werent retrieved

- Optional colpali reranking
- Pass all retrieved documents through colpali to get more finegrained embeddings of content ()
- Then use chunked text + VLM on figures and graphs to pass through to generator context window

## 5. Evals
Evaluation is automated via a RAGAS harness that runs on every push to main through GitHub Actions, using a self-hosted runner on the workstation to access the RTX 3090 and local services.

**Metrics tracked:**

- `faithfulness` whether answer is based on retrieved context
- `answer_relevancy`, whether answer addresses query
- `context_precision` whether retrieved chunks are relevant to query

- `evals/test_dataset.json` contains QA pairs with ground truth answers drawn from the ingested scientific literature
- `evals/run_eval.py` pushes each question through the full inference pipeline (`RunInference`) and collects the query, retrieved contexts, generated answer, and ground truth
- RAGAS scores the collected traces using a local judge LLM (vLLM + Qwen3-32B-AWQ) and embedding model (BAAI/bge-m3)
