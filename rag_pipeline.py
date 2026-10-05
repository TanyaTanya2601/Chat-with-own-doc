from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def build_chain(
    pdf_path: str,
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    top_k: int = 4,
) -> ConversationalRetrievalChain:
   
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=api_key
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatGoogleGenerativeAI(
        model=model_name, google_api_key=api_key, temperature=0
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a question against a built chain.
    """
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "source_documents": result.get("source_documents", []),
    }