import asyncio
import math
import os
import re
import ssl

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
)
index = os.getenv("VECTOR_STORE_INDEX_NAME")
vector_store = PineconeVectorStore(index_name=index, embedding=embeddings)


async def index_documents_async(documents: list[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch_document, batch_num: int):
        try:
            await vector_store.aadd_documents(batch_document)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch_document)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    results = []
    for i, batch in enumerate(batches):
        res = await add_batch(batch, i + 1)
        results.append(res)

    print(f"Indexing Results {results}")

    successful_batches = sum([1 for res in results if res == True])
    failed_batches = sum([1 for res in results if res == False])

    if successful_batches == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful_batches}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful_batches}/{len(batches)} batches successfully \n Failed {failed_batches}/{len(batches)} "
        )


def parse_llms_txt(url):
    text = requests.get(url).text

    pattern = r"- \[(.*?)\]\((.*?)\)"
    matches = re.findall(pattern, text)

    return [{"title": title, "url": link} for title, link in matches]


def fetch_md(url):
    return requests.get(url).text


def build_chunk_docs(entries, splitter: RecursiveCharacterTextSplitter):
    docs = []

    for e in entries:
        md = fetch_md(e["url"])
        if not md:
            continue

        chunks = splitter.split_text(md)

        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"title": e["title"], "source": e["url"], "chunk_id": i},
                )
            )

    return docs


async def ingest():
    """Main async function to orchestrate the entire process."""
    log_header(" INGESTION PIPELINE")

    log_info(
        "🗺️   Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    doc_link = "https://docs.langchain.com/llms.txt"

    urlMap = parse_llms_txt(doc_link)
    urlMap = urlMap[0 : math.floor(len(urlMap) / 4)]

    log_header("DOCUMENT CHUNKING PHASE")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    all_docs = build_chunk_docs(urlMap, text_splitter)

    log_success(f"Text Splitter: Created {len(all_docs)} chunks  documents")

    await index_documents_async(documents=all_docs, batch_size=100)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(all_docs)}")


if __name__ == "__main__":
    asyncio.run(ingest())
