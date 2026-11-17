import os
import pprint
import getpass
import bs4
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


import gradio as gr

load_dotenv()

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
def get_sitemap(url):
    req = Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response = urlopen(req)
    xml = BeautifulSoup(
        response,
        "lxml-xml",
        from_encoding=response.info().get_param("charset")
    )
    return xml

def get_urls(xml, name=None, data=None, verbose=False):
    urls = []
    for url in xml.find_all("url"):
        if xml.find("loc"):
            loc = url.findNext("loc").text
            urls.append(loc)
    return urls

url = "https://zerodha.com/varsity/chapter-sitemap2.xml"
xml = get_sitemap(url)
urls = get_urls(xml, verbose=False)

docs = []
# Loading just the first 10 for speed; adjust as needed
for i, url in enumerate(urls[:10]):
    loader = WebBaseLoader(url)
    docs.extend(loader.load())
    if i % 10 == 0:
        print("Loaded document index:", i)


#####################################################
#       2. Vectore Store 
#####################################################
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}, search_type = 'similarity')

#####################################################
#       3. RAG CHAIN SETUP
#####################################################

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

#create context_query_prompt to build standalone question
context_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, context_query_prompt)

#system_prompt
system_prompt = """
You are a helpful assistant that can answer questions about the financial markets.
You are given a chat history and the latest user question which might reference context in the chat history.
Retrived relevant context from the vector store and answer the question.
If the question is not related to the context, say "I don't know".
\n\n
Context : {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, answer_chain)


########################################################
#       4. CHATBOT SETUP
#####################################################

def predict(message, history):
    """
    message: str
    history: list of dicts in OpenAI-style format when type='messages'
             e.g. [{"role":"user","content":"hi"}, {"role":"assistant","content":"hello"}]
    """
    history_for_llm = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            history_for_llm.append(HumanMessage(content=content))
        elif role == "assistant":
            history_for_llm.append(AIMessage(content=content))

    result = rag_chain.invoke({"input": message, "history": history_for_llm})
    return result["answer"]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# DocumentQABot")

    chatbot = gr.Chatbot(height=400, type="messages", allow_tags=False)
    msg = gr.Textbox(
        placeholder="Hi! I am your virtual assistant, how can I help you today?",
        container=False,
        scale=7,
    )

    with gr.Row():
        undo = gr.Button("Delete Previous")
        clear = gr.Button("Clear")

    chat = gr.ChatInterface(
        fn=predict,
        chatbot=chatbot,
        textbox=msg,
        examples=["What is index fund?", "Where to buy stocks?"],
        title=None,  # already using Markdown title
    )

    # Clear chat
    clear.click(lambda: [], None, chatbot)

    # Undo last turn (removes last 2 messages if present: user+assistant)
    def undo_last(history):
        if not history:
            return []
        # In messages format, history is a list of dicts, typically alternating user/assistant
        return history[:-2] if len(history) >= 2 else []

    undo.click(undo_last, chatbot, chatbot)

demo.launch(share=True, debug=False)
