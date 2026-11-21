from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from snackstack.logger import setup_logger
from snackstack.data import MENU

#─── Logger ───────────────────────────────────────────────────
logger = setup_logger("rag")

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "snackstack_db")
COLLECTION_NAME = "snackstack_menu"

#─────OpenAI Embeddings ───────────────────────────────────────────────────
# check_embedding_ctx_length=False: skips tiktoken, which downloads a tokenizer file
# on first use (fails behind corporate proxies). Our menu texts are ~80 tokens, far below the 8k limit.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", check_embedding_ctx_length=False)

#───get menu  items ───────────────────────────────────────────────────
def get_menu_items() -> list[dict]:
    """Get the menu items from the data --> menu. This function is used to get the menu items from the data.

    Returns:
        list[Document]: A list of documents containing the menu items.
    """
    docs: list[Document] = []

    for menu_item in MENU:
        content = (
            f"id : {menu_item['id']}\n"
            f"name : {menu_item['name']}\n"
            f"category : {menu_item['category']}\n"
            f"cuisine : {menu_item['cuisine']}\n"
            f"price : {menu_item['price']}\n"
            f"rating : {menu_item['rating']}\n"
            f"dietary_tags : {', '.join(menu_item['dietary_tags']) or 'None'}\n"
            f"description : {menu_item['description']}\n"
            f"availability : {menu_item['availability']}\n"
        )
        docs.append(
            Document(
                id=menu_item['id'],
                page_content=content,
                metadata={
                    "id": menu_item['id'],
                    "name": menu_item['name'],
                    "category": menu_item['category'],
                    "cuisine": menu_item['cuisine'],
                    "price": menu_item['price'],
                    "rating": menu_item['rating'],
                    # FIX: Chroma rejects EMPTY lists (Classic Cheeseburger has []) -> store as a string.
                    "dietary_tags": ", ".join(menu_item['dietary_tags']) or "None",
                },
            )
        )
    return docs



#───get menu items ───────────────────────────────────────────────────
get_menu = get_menu_items()


#───splitter ───────────────────────────────────────────────────
def doc_splitter(docs: list[Document]) -> list[Document]:
    """Split the documents into chunks. This function is used to split the documents into chunks.

    Args:
        docs: list[Document]: A list of documents to split.

    Returns:
        list[Document]: A list of documents containing the chunks.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
    logger.info(f"Splitting {len(docs)} documents into chunks")
    return splitter.split_documents(docs)

def exist_vectorstore(path: str = DB_PATH, collection_name: str = COLLECTION_NAME) -> Chroma | None:
    """Return the persisted vector store if it already contains documents, else None."""

    vector_store = Chroma(persist_directory = path, embedding_function=embeddings, collection_name= collection_name)

    # FIX: a Chroma object is always truthy, so `if not vector_store` never rebuilt
    # and search ran on an EMPTY collection. Check the document count instead.
    if vector_store._collection.count() == 0:
        return None
    logger.info(f"Vector store exists at {path}")
    return vector_store

def build_vectorstore() -> Chroma:
    """Build the vector store from the data --> menu and orders. This function is used to build the vector store from the data.

    Returns:
        Chroma: A vector store containing the menu and orders.
    """
    menu_docs = doc_splitter(get_menu)
    logger.info(f"Building vector store with {len(menu_docs)} documents")

    vectorstore = Chroma.from_documents(
    documents=menu_docs,
    embedding=embeddings,
    ids=[f"{d.metadata['id']}-{i}" for i, d in enumerate(menu_docs)],  # stable ids -> no duplicates
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
    )
    logger.info(f"Vector store built successfully")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Get the vector store from the data --> menu and orders. This function is used to get the vector store from the data.

    Returns:
        Chroma: A vector store containing the menu and orders.
    """
    vector_store =  exist_vectorstore()

    if not vector_store:
        vector_store = build_vectorstore()
   
    return vector_store


def create_retriver(vector_store: Chroma, k: int = 4) -> VectorStoreRetriever:
    """Create a retriever from the vector store. This function is used to create a retriever from the vector store.

    Args:
        vector_store: Chroma: A vector store containing the menu and orders.

    Returns:
        Retriever: A retriever containing the menu and orders.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})


#──────── vector store and retriever to be called by the agents / graph───────────────────────────────────────────────────
vector_store = get_vectorstore()
retriever = create_retriver(vector_store)
