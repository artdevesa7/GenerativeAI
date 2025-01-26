from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Load and Prepare Documents
def prepare_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs

# Step 2: Create Vector Store
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Step 3: Create RetrievalQA Chain
def create_retrieval_qa(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Step 4: Define Tools
def define_tools(qa_chain):
    tools = [
        Tool(
            name="Document QA",
            func=qa_chain.run,
            description="Use this tool to answer questions based on document retrieval."
        )
    ]
    return tools

# Step 5: Initialize Agent
def initialize_ai_agent(tools):
    llm = OpenAI(temperature=0)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    return agent

# Main Workflow
if __name__ == "__main__":
    # Prepare documents and vector store
    docs = prepare_documents("your_documents.txt")
    vectorstore = create_vector_store(docs)

    # Create RetrievalQA and tools
    qa_chain = create_retrieval_qa(vectorstore)
    tools = define_tools(qa_chain)

    # Initialize the agent
    agent = initialize_ai_agent(tools)

    # Use the agent
    while True:
        query = input("Ask your question: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting the AI Agent. Goodbye!")
            break
        response = agent.run(query)
        print("\nResponse:", response)
