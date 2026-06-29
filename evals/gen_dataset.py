import json
import pandas as pd
from datasets import Dataset
from openai import OpenAI
from pathlib import Path
import argparse
import random

#from ragas.metrics import context_precision, faithfulness, answer_relevancy
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.synthesizers import (SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer)
from ragas.run_config import RunConfig
from ragas.testset.transforms import Parallel
from ragas.testset.transforms import KeyphrasesExtractor, SummaryExtractor, CosineSimilarityBuilder, Parallel, EmbeddingExtractor, TitleExtractor, HeadlinesExtractor

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import config

def load_md_files(base_path):
    md_docs = []                                                
    file_pattern = "*/auto/*.md"                                
    md_files = list(base_path.glob(file_pattern))               

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        filename = md_path.name
        
        doc = Document(
            page_content=content,
            metadata={
                "source": filename,
                "file_name": f"{filename}.md",
                "abs_path": str(md_path.resolve())
            }
        )
        
        md_docs.append(doc)

    print(f"Loaded {len(md_docs)} documents.")
    return md_docs

def generate_dataset(mode):                                                         # single, multi abstract, multi specific

    # Paths for markdown docs
    md_path = Path(config.mineru.host_output_dir)                             # mineru-output dir
    md_docs = load_md_files(md_path)

    # ----------------------------

    # Setup vllm api and embeddings for ragas
    vllm_chat = ChatOpenAI(
        model=config.models.vlm_model,               # using vlm to match inferncne 
        base_url=config.network.vllm_api_base,       
        api_key="EMPTY",                            
        temperature=0.0,                              # 0.4, 0.5 for test
        max_tokens=512,
        max_retries=2,)
        #model_kwargs={"response_format": {"type": "json_object"}}

    llm_wrapper = LangchainLLMWrapper(vllm_chat)

    embeddings_model = HuggingFaceEmbeddings(
        model_name=config.models.embedding_model,    
        model_kwargs={"device": "cuda"}              
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings_model)

    # --------------------------    

    # Fast splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    docs = splitter.split_documents(md_docs)
    docs = random.sample(docs, min(20, len(docs)))
    print(f"Number of docs {len(docs)}")
    '''
    # Specify limited extractors (no headlines, title etc)
    transforms = [Parallel(SummaryExtractor(llm=llm_wrapper), KeyphrasesExtractor(llm=llm_wrapper),),
        EmbeddingExtractor(embedding_model=embeddings_wrapper),
        CosineSimilarityBuilder(),]
    '''

    transforms = [
        Parallel(
            TitleExtractor(llm=llm_wrapper),
            HeadlinesExtractor(llm=llm_wrapper),
            SummaryExtractor(llm=llm_wrapper), 
            KeyphrasesExtractor(llm=llm_wrapper)
        ),
        EmbeddingExtractor(embedding_model=embeddings_wrapper),
        CosineSimilarityBuilder(),
    ]
    # ------------------------

    # Specify distribution for single vs multi hop dataset
    if mode == 'single':
        distribution = [(SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 1.0)]
        target_size = 5                                                             # for smoke
    else:
        distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.3),
            (MultiHopAbstractQuerySynthesizer(llm=llm_wrapper), 0.7),
        ]
        target_size = 10

    # ---------------------------
    run_config = RunConfig(timeout=180, max_workers=8)

    # create generator then gen datasets
    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings_wrapper)
    dataset = generator.generate_with_langchain_docs(docs, 
                                                    testset_size=target_size, 
                                                    query_distribution=distribution, 
                                                    run_config=run_config,
                                                    transforms=transforms)

    # save to evals folder
    output_dir = Path(config.DirConfig.evals) / "generated_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = dataset.to_pandas()
    df.to_json(output_dir / "testset.json", orient="records", indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "full"], default="single")
    args = parser.parse_args()
    
    generate_dataset(mode=args.mode)

if __name__ == "__main__":
    main()