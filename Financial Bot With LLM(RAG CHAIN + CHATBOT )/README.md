# Financial Bot with LLM (RAG Chain + Chatbot)

## Overview
This project is a chatbot that answers questions about financial markets. It uses Retrieval-Augmented Generation (RAG): it pulls the relevant passages from Zerodha Varsity's learning content and writes its answer from them. It remembers earlier messages in the chat, so you can ask follow-up questions.

## What Was Built
1. **Data loading:** The bot reads the Zerodha Varsity chapter sitemap, collects the chapter URLs and loads their web pages. It currently loads the first 10 chapters.
2. **Vector store:** The pages are split into chunks of 1,000 characters, with 200 characters of overlap between chunks. Each chunk is converted to an embedding with OpenAI's `text-embedding-3-small` and stored in a Chroma vector database.
3. **History-aware retriever:** The LLM rewrites each new question as a standalone question using the chat history. The bot then fetches the 3 most relevant chunks.
4. **RAG chain:** The retrieved chunks, the chat history and the question go to `gpt-4o-mini`, which answers using only that content. If the question is outside the content, the bot replies "I don't know".
5. **Chat interface:** A Gradio web UI with a chat window, example questions, and buttons to clear the chat or delete the last exchange.

## Modules Used
| Module | Purpose |
|---|---|
| `langchain-openai` | OpenAI chat model (`gpt-4o-mini`) and embeddings |
| `langchain-community` | `WebBaseLoader` to load web pages |
| `langchain-text-splitters` | Splits documents into chunks |
| `langchain-chroma` | Chroma vector store for semantic search |
| `langchain-core` | Prompt templates and message types |
| `langchain-classic` | Retrieval chain, history-aware retriever and document-combining chain |
| `beautifulsoup4`, `lxml` | Parses the XML sitemap |
| `python-dotenv` | Loads the OpenAI API key from `.env` |
| `gradio` | Web chat interface |

## Setup
1. Install the dependencies:
   ```bash
   pip install langchain-openai langchain-community langchain-text-splitters langchain-chroma langchain-core langchain-classic beautifulsoup4 lxml python-dotenv gradio
   ```
2. Create a `.env` file in the project folder:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Run the bot:
   ```bash
   python financial_bot.py
   ```

## End Result
A web chatbot that runs locally and also gives you a public link. It answers financial-market questions such as "What is an index fund?" using Zerodha Varsity content, and it handles follow-up questions within the same conversation.