import streamlit as st
import groq
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.tools import Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
import warnings

warnings.simplefilter("ignore")

# Load environment variables from .env file
load_dotenv()

# Set up Groq client
groq_client = groq.Client(api_key=os.environ["GROQ_API_KEY"])

# Set up LangChain Groq LLM
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"],
               model_name="mixtral-8x7b-32768",
               temperature=0.1,
               max_tokens=32000,
               max_retries=3)

# Set up memory
memory = ConversationBufferMemory(return_messages=True)

# Streamlit UI
st.title("Groq Chatbot with Document RAG and Internet Search")

# Document processing progress
doc_progress = st.progress(0)
st.write("Processing documents...")

VECTORSTORE_PATH = "./vectorstore"

# Set up embeddings
embeddings = HuggingFaceEmbeddings()

# Check if the vector store already exists
if os.path.exists(f"{VECTORSTORE_PATH}.faiss"):
    print("Loading existing vector store...")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
    doc_progress.progress(100)
    print("Vector store loaded successfully")
else:
    # Load documents
    print("Loading documents...")
    loader = DirectoryLoader("./documents", glob="**/*.pdf")
    documents = loader.load()
    doc_progress.progress(25)
    print(f"Loaded {len(documents)} documents")

    # Split documents
    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)
    doc_progress.progress(50)
    print(f"Split into {len(texts)} text chunks")

    # Create embeddings and vector store
    print("Creating embeddings and vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    doc_progress.progress(75)
    print("Vector store created and saved")

    # Create retriever
    retriever = vectorstore.as_retriever()
    doc_progress.progress(100)
    print("Document processing complete")

# Set up SerpAPI
search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"])

# Set up tools
tools = [
    Tool(
        name="Document Search",
        func=retriever.get_relevant_documents,
        description="Useful for searching information in the loaded documents"
    ),
    Tool(
        name="Internet Search",
        func=search.run,
        description="Useful for searching the internet for current information"
    )
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

st.write("Document processing complete. Ready for chat!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use the agent to generate a response
        agent_response = agent.run(prompt)

        for response in groq_client.chat.completions.create(
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ] + [{"role": "assistant", "content": agent_response}],
            model="mixtral-8x7b-32768",
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
