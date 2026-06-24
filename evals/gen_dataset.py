import json
import pandas as pd
from datasets import Dataset
from openai import OpenAI
from pathlib import Path
import argparse

from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.synthesizers import (SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer)

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from src.inference import InferencePipeline 
from src import config
from src.clients import embedding_client, vlm_client

def load_md_files(base_path):

    md_docs = []                                                # store all .md docs
    file_pattern = "*/auto/*.md"                                # mineru-output/filename/auto/filename.md
    md_files = list(base_path.glob(file_pattern))               # find paths of md files

    # iterate through md files, create langchain document for each
    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        filename = md_path.name

        doc = Document(page_content=content, metadata={"source": filename,
                                                       "file_name": f"{filename}.md",
                                                       "abs_path": str((md_path).resolve())
                                                       })   
        md_docs.append(doc)

    print(len(md_docs))
    return md_docs

def generate_dataset(mode):                                                         # single, multi abstract, multi specific

    # Paths for markdown docs
    md_path = Path(config.MineruConfig.host_output_dir)                             # mineru-output dir
    md_docs = load_md_files(md_path)

    # ----------------------------

    # Setup vllm api and embeddings for ragas
    vllm_chat = ChatOpenAI(
        model=config.models.vlm_model,               # using vlm to match inferncne 
        base_url=config.network.vllm_api_base,       
        api_key="EMPTY",                            
        temperature=0.0                              # 0.4, 0.5 for test
    )
    llm_wrapper = LangchainLLMWrapper(vllm_chat)

    embeddings_model = HuggingFaceEmbeddings(
        model_name=config.models.embedding_model,    
        model_kwargs={"device": "cpu"}              
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings_model)

    # --------------------------    

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

    # create generator then gen datasets
    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings_wrapper)
    dataset = generator.generate_with_langchain_docs(md_docs, testset_size=target_size, query_distribution=distribution)

    # save to evals folder
    output_path = Path(config.DirConfig.evals) / "generated_datasets"
    dataset.to_json(output_path, orient="records", indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "full"], default="single")
    args = parser.parse_args()
    
    generate_dataset(mode=args.mode)

if __name__ == "__main__":
    main()