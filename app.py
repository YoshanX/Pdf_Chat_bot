import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- Configuration ---
# Local embedding model (runs locally on CPU, no API key needed for this part)
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# GENERATIVE LLM for synthesis and query transformation (Requires HF_TOKEN)
QA_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2" 

# --- Helper Functions ---

@st.cache_resource
def initialize_embedding_model():
    """Initializes and caches the local HuggingFace embedding model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={'device': 'cpu'} 
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing local embedding model: {e}")
        st.stop()
        
@st.cache_resource
def load_pdf_and_create_vector_store(pdf_content_bytes):
    """
    Loads PDF, splits text, retrieves the cached embeddings model, 
    and creates the FAISS vector store.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_content_bytes)
        pdf_path = tmp_file.name

    try:
        embeddings = initialize_embedding_model() 
        loader = PyPDFLoader(pdf_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(data)

        with st.spinner("Processing document and creating knowledge base..."):
            db = FAISS.from_documents(docs, embeddings) 
        
        st.success("Knowledge base created successfully! You can now ask questions.")
        return db
        
    finally:
        os.unlink(pdf_path)

def format_chat_history(history):
    """
    Formats st.session_state.chat_history into a clean string for the LLM.
    """
    formatted = []
    for msg in history:
        if msg["role"] == "user":
            formatted.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            # Safely get the answer text from the dictionary content
            answer_text = msg['content'].get('answer')
            if answer_text:
                formatted.append(f"Assistant: {answer_text}")
    
    # Limit to the last few exchanges to save tokens
    return "\n".join(formatted[-6:])


def get_pdf_answer(question: str, db: FAISS, qa_client: InferenceClient, chat_history: list):
    """
    Executes the advanced RAG pipeline with History-Aware Retrieval (HAR)
    and returns the answer along with citation sources.
    """
    
    formatted_history = format_chat_history(chat_history)

    # --- LLM Call 1: Query Transformation (HAR) ---
    transformation_system = "You are a query rewriter. Given the conversation history and the latest user question, generate a concise, standalone search query that captures the full context of the user's intent. Do NOT answer the question. Your output must be only the standalone query."
    transformation_prompt = f"Chat History:\n{formatted_history}\n\nLatest Question: {question}\n\nStandalone Search Query:"

    transformation_messages = [
        {"role": "system", "content": transformation_system},
        {"role": "user", "content": transformation_prompt}
    ]
    
    try:
        transformation_response = qa_client.chat_completion(
            model=QA_MODEL_ID,
            messages=transformation_messages,
            max_tokens=100,
            temperature=0.0
        )
        standalone_query = transformation_response.choices[0].message.content.strip()
        
        if not standalone_query or len(standalone_query.split()) < 3:
            standalone_query = question

    except Exception as e:
        standalone_query = question
        st.warning(f"Query rewriting failed ({e}). Searching using original question.")

    # 2. Retrieve relevant context using the standalone query
    retrieved_docs = db.similarity_search(standalone_query, k=3)
    
    # Process retrieved documents to build context and citation list
    context = ""
    citations = []
    for i, doc in enumerate(retrieved_docs):
        context += doc.page_content + "\n\n"
        page_num = doc.metadata.get('page', 'Unknown')
        
        citations.append({
            "source": doc.metadata.get('source', 'Uploaded Document'),
            "page": int(page_num) + 1, 
            "snippet": doc.page_content[:150].replace('\n', ' ') + "..." 
        })
    
    st.session_state.chat_history.append({"role": "context", "content": context})
    
    # 3. LLM Call 2: Final Answer Generation (RAG)
    system_instruction = "You are an intelligent, helpful, and concise assistant. Your task is to answer the user's question based ONLY on the CONTEXT provided below. If the user asks a conversational question (like 'hello'), reply appropriately. If the user asks about the document, synthesize the answer conversationally. If the answer is not present in the context, politely state that the information could not be found in the document."
    
    user_prompt = f"CONTEXT:\n---\n{context}\n---\n\nUSER QUESTION: {question}"

    final_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = qa_client.chat_completion(
            model=QA_MODEL_ID,
            messages=final_messages,
            max_tokens=256,
            temperature=0.1, 
        )
        answer = response.choices[0].message.content.strip()
        
        return answer, citations
        
    except Exception as e:
        return f"Error connecting to Hugging Face LLM: Please check your API Token or model access. Error: {e}", []

# Clear chat history function
def clear_chat_history():
    """Clears the chat history in the session state."""
    st.session_state.chat_history = []

# --- Main Streamlit App ---

def main():
    # Load environment variables from .env file
    load_dotenv()
    hf_token = os.getenv("api_key") 
    
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 PDF Question Answering Bot")
    st.markdown("I can answer follow-up questions based on your uploaded document.")
    
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db" not in st.session_state:
        st.session_state.db = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    # --- Sidebar for Setup (File Upload) ---
    with st.sidebar:
        st.header("Setup & Configuration")
        
        # 1. API Key Status (Simplified, no input widget)
        if hf_token:
            st.success("Setup & Configuration successfull")
        else:
            st.error("FATAL: API Key (`api_key`) not found in .env file.")
            st.stop()

        # 2. PDF File Uploader
        uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
        
        # Logic to check if a new file needs processing
        if uploaded_file and hf_token:
            process_file = (uploaded_file.name != st.session_state.uploaded_file_name)
            
            if process_file:
                st.session_state.db = None
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.chat_history = [] # Clear chat for new document
                
            if process_file or st.session_state.db is None:
                st.session_state.db = load_pdf_and_create_vector_store(uploaded_file.getvalue())
                st.session_state.qa_client = InferenceClient(token=hf_token)
                st.success("Setup complete. Ask your question below!")
        
        elif not uploaded_file:
            st.warning("Please upload a PDF document to begin.")
        elif st.session_state.db:
             st.success(f"Ready to chat with: {st.session_state.uploaded_file_name}")

        st.markdown("---") 
        # 3. Clear Chat Button
        st.button("🧹 Clear Chat History", on_click=clear_chat_history)
        st.caption("Resets the conversation in the main area.")


    # --- Main Chat Area ---
    
    # Display historical messages 
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            # Assistant message content is a dictionary {answer: str, citations: list}
            content = message["content"]
            
            with st.chat_message("assistant"):
                # Display the main answer
                st.markdown(content['answer']) 
                
                # Immediately display citations in the same message block 
                citations = content.get('citations', [])
                if citations:
                    with st.expander("📚 Sources Used for Answer"):
                        for citation in citations:
                            st.caption(f"**Source:** {citation['source']} (Page {citation['page']})")
                            st.code(citation['snippet'], language="markdown")


    # Handle user input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        
        # --- FIXED: Append user message to history IMMEDIATELY ---
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # --------------------------------------------------------

        # 1. Display user message immediately (INSTANT VISUAL FEEDBACK)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. LLM RAG/Conversational Pipeline (Runs while spinner is active)
        
        if not st.session_state.db or not hasattr(st.session_state, 'qa_client'):
            formatted_response = "I can't answer questions yet. Please complete the setup in the sidebar by uploading a PDF."
            response_content = {"answer": formatted_response, "citations": []}
        
        else:
            # RAG pipeline runs, showing the spinner
            with st.spinner("Analyzing history, searching document, and generating answer..."):
                answer, citations = get_pdf_answer(
                    question=prompt, 
                    db=st.session_state.db, 
                    qa_client=st.session_state.qa_client,
                    chat_history=st.session_state.chat_history
                )
                
            # 3. Prepare content for state
            response_content = {"answer": answer, "citations": citations}
            
        # 4. Append the final response content to state
        st.session_state.chat_history.append({"role": "assistant", "content": response_content})
        
        # 5. Force the script to rerun (CRITICAL)
        st.rerun()


if __name__ == "__main__":
    main()