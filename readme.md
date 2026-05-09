# Multi-modal RAG for scientific literature QA

Purpose of this project is to develop a pipeline for large scale scientific paper processing and QA. The intent is to incoroporate this into an autonomous agentic research agent. 

# Progress
| Stage                                            | Status        |
|--------------------------------------------------|---------------|
| PDF ingestion (MinerU)                           | Complete      |
| Hybrid chunking + contextual prefixing           | Complete      |
| Graph construction (entity & relation extraction)| Complete      |
| Multimodal block handling (figures, equations)   | In progress   |
| Retrieval pipeline                               | In progress   |
| Generation (vLLM + Qwen3-VL)                     | In progress   |
| ColPali visual retrieval                         | Planned       |
| RAGAS evaluation harness                         | Planned       |


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

- Optional colapli reranking
- Pass all retrieved documents through colpali to get more finegrained embeddings of content ()
- Then use chunked text + VLM on figures and graphs to pass through to generator context window

## 5. Evals
- TODO
- 

## Design Considerations
Due to the complexity of conducting open ended research on a wide variety of topics over a large number of documents, a GraphRAG approach was chosen over standard RAG giving it the ability to perform multi-step reasoning between various locations in a document and between documents. LLM model
Gemma 3 30B AWQ model was chosen for all of the LLM querying functionality due to balancing its native multi-modal capabilities and also GPU resource constraints. 
