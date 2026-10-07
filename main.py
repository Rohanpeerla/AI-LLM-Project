import os
import time
import streamlit as st
from dotenv import load_dotenv

# LangChain & Gemini Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQAWithSourcesChain

# Load environment variables from .env file
load_dotenv()
os.environ["USER_AGENT"] = "RockyBotNewsApp/1.0"

# Streamlit Page Setup
st.set_page_config(
    page_title="RockyBot: News Research Tool (Gemini)",
    page_icon="📈",
    layout="wide"
)

st.title("RockyBot: News Research Tool 📈 (Powered by Gemini)")
st.caption("Load news article URLs, process them into vector storage, and ask research questions with source citations!")

# Check API Key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# Sidebar Configuration
st.sidebar.title("Configuration & Inputs")

if not api_key:
    api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password", help="Get yours at https://aistudio.google.com/")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

st.sidebar.subheader("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs 🚀", use_container_width=True)
DB_FAISS_PATH = "faiss_store_gemini"

main_placeholder = st.empty()

# Cached Function to Load FAISS Index & Chain for fast response
@st.cache_resource(show_spinner=False)
def get_qa_chain(_api_key: str, db_path: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=_api_key
    )
    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_api_key,
        temperature=0.4,
        max_output_tokens=600
    )
    return RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

# Processing Pipeline
if process_url_clicked:
    valid_urls = [u.strip() for u in urls if u.strip()]
    
    if not valid_urls:
        st.error("Please enter at least one valid URL!")
    elif not api_key:
        st.error("Google Gemini API Key is missing! Please provide it in the sidebar or in your `.env` file.")
    else:
        try:
            # Clear previous cache when processing new URLs
            st.cache_resource.clear()

            # 1. Load Data from URLs using WebBaseLoader (BeautifulSoup)
            main_placeholder.text("Data Loading... Started... ⏳")
            loader = WebBaseLoader(
                web_paths=valid_urls,
                header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            data = loader.load()
            
            if not data:
                main_placeholder.error("Could not fetch content from the provided URLs. Please check the URLs and try again.")
            else:
                # 2. Split Text into Chunks
                main_placeholder.text("Text Splitting... Started... ✂️")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=800,
                    chunk_overlap=150
                )
                docs = text_splitter.split_documents(data)
                
                # 3. Create Embeddings & FAISS Vector Store
                main_placeholder.text("Building Vector Index with Gemini Embeddings... 🧠")
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=api_key
                )
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # 4. Save Vector Store Locally
                vectorstore.save_local(DB_FAISS_PATH)
                main_placeholder.success("Processing Complete! Index saved successfully. ✅")
                time.sleep(1.5)
                main_placeholder.empty()

        except Exception as e:
            main_placeholder.error(f"An error occurred during processing: {str(e)}")

# Question & Answering Section
st.divider()
st.subheader("Ask Questions about the News Articles")

query = st.text_input("Question:", placeholder="e.g., What are the main key takeaways or financial impacts mentioned?")

if query:
    if not api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar.")
    elif not os.path.exists(DB_FAISS_PATH):
        st.warning("No vector database found! Please process URLs first using the sidebar.")
    else:
        with st.spinner("Analyzing articles with Gemini LLM... 🔍"):
            try:
                chain = get_qa_chain(api_key, DB_FAISS_PATH)
                result = chain.invoke({"question": query})
                
                st.markdown("### 📝 Answer")
                st.write(result.get("answer", "No answer generated."))
                
                sources = result.get("sources", "")
                if sources and sources.strip():
                    st.markdown("### 🔗 Sources")
                    sources_list = [s.strip() for s in sources.split("\n") if s.strip()]
                    for source in sources_list:
                        st.markdown(f"- [{source}]({source})" if source.startswith("http") else f"- {source}")

            except Exception as e:
                st.error(f"Error answering question: {str(e)}")
