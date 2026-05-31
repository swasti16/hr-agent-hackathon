"""
RAG Pipeline — Core Implementation

Builds a RAG pipeline over policy documents.
Used as the system under test for all RAG evaluation tests.

Pipeline:
  1. Load documents
  2. Split into chunks
  3. Embed using HuggingFace sentence-transformers
  4. Store in ChromaDB vector store
  5. Query: retrieve top-k chunks + generate answer via LLM
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.settings import settings
from src.utils.llm_factory import get_llm, get_vector_store

# ======== Pipeline Class ==================================


class RagPipeline:
    """
    RAG Pipeline.
    Loads documents, indexes them, and answers questions.
    """

    def __init__(self, docs_dir: str | None = None,
                 collection_name: str | None = None,
                 system_prompt: str | None = None,
                 persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.docs_dir = docs_dir
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.vector_store = get_vector_store(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir)
        self.retriever = None
        self.chain = None
        self.llm = get_llm(temperature=0.1)
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self._setup_retriever_and_chain()

    def split_into_chunks(self, documents: list) -> list:
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(documents)

    def _index_documents(self, loader: DirectoryLoader) -> int:
        """Split documents into chunks and store in ChromaDB."""
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        chunks = self.split_into_chunks(documents)
        print(f"Split into {len(chunks)} chunks")

        self.vector_store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks into ChromaDB")

        return len(chunks)

    def _setup_retriever_and_chain(self) -> None:
        """
        Always call this after vector_store changes.
        Updates retriever and chain to point to current vector_store.
        """
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.TOP_K}
        )
        self._build_chain()

    def load_and_index(self, force_reindex=False) -> int:
        """
        Load documents, chunk them, and index into ChromaDB.
        Returns number of chunks indexed.
        """
        existing_count = self.get_chunk_count()
        print(f"Existing chunks in DB before indexing: {existing_count}")
        if not force_reindex and existing_count > 0:
            print("Loaded existing ChromaDB index")
            return existing_count
        if existing_count > 0:
            print(f"Found {existing_count} existing chunks. "
                  "Deleting for fresh indexing...")
        else:
            print("Fresh database. Starting indexing...")
        try:
            self.vector_store.delete_collection()
        except Exception as e:
            print(f"Warning: Could not delete existing collection. "
                  f"Proceeding with indexing. Error: {e}")
        self.vector_store = get_vector_store(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir)
        print(f"Loading documents from {self.docs_dir}...")

        # Load all .txt files from hr_documents folder
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        chunks_len = self._index_documents(loader)
        self._setup_retriever_and_chain()
        return chunks_len

    def _build_chain(self) -> None:
        """Build the RAG chain: retrieve + generate."""

        # Prompt template with system instructions and retrieved context
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.system_prompt}\n\nContext :{{context}}"),
            ("human", "{question}")
        ])

        def format_docs(docs):
            """Format retrieved documents into a single string for the prompt."""
            return "\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> dict:
        """
        Ask a question and get answer with retrieved context.

        Returns:
            {
                "question": str,
                "answer": str,
                "contexts": list[str],
                "sources": list[str]
            }
        """
        if not self.chain:
            raise RuntimeError(
                "Pipeline chain not built. "
                "This should not happen - check __init__."
            )

        # Get answer
        answer = self.chain.invoke(question)

        # Get retrieved chunks (for RAGAS evaluation)
        retrieved_docs = self.retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        sources = [
            doc.metadata.get("source", "unknown")
            for doc in retrieved_docs
        ]

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "sources": sources
        }

    def get_chunk_count(self) -> int:
        """Returns total number of chunks in vector store."""
        if not self.vector_store:
            return 0
        collection = self.vector_store._collection
        return collection.count()

    def add_new_document(self, file_path: str) -> int:
        """
        Add a SINGLE new document without re-indexing everything.
        Production approach — efficient, no duplication.
        """
        # Load only the new document
        loader = TextLoader(file_path, encoding="utf-8")
        chunks_len = self._index_documents(loader)
        self._setup_retriever_and_chain()
        return chunks_len
