import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
model = init_chat_model(model="gpt-5.5")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
index = os.getenv("VECTOR_STORE_INDEX_NAME")
vector_store = PineconeVectorStore(index_name=index, embedding=embeddings)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]


prompt = SystemMessage(content="""
        You are a helpful AI assistant that answers questions about LangChain documentation. \n
        You have access to a tool that retrieves relevant documentation. \n
        Use the tool to find relevant information before answering questions. \n
        Always cite the sources you use in your answers. \n
        If you cannot find the answer in the retrieved documentation, say so. \n
         """)


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    agent = create_agent(model, tools, system_prompt=prompt)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    answer = result["messages"][-1].content

    print(answer)
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in result["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    print("Retrieving User Query")
    run_llm("what is an Agent ? How does it work?")
