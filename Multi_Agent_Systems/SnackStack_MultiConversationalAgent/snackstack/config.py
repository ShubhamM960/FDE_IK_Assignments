import os
import getpass
import sys
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from snackstack.logger import setup_logger

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

#── Logger ───────────────────────────────────────────────────
logger = setup_logger("config")


#── Get OpenAI API key ───────────────────────────────────────────────────
def get_openai_key(key : str) :
    api_key = os.getenv(key)
    if not api_key :
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ[key] = api_key
    return api_key

api_key = get_openai_key("OPENAI_API_KEY")


#── OpenAI ───────────────────────────────────────────────────
openai_client = OpenAI(api_key=api_key)

#── LLM ───────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)
