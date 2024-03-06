import json
import logging
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Document
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

logging.basicConfig(stream=sys.stdout, level="DEBUG")

app = FastAPI()

@app.get("/load/{id}")
async def load(id: int):
    documents = load_documents()
    # service_context, token_counter = _setup_service_context()
    token_counter, callback_manager = _setup_token_counter()

    # index = VectorStoreIndex.from_documents([Document(text=documents[id]["text"])], service_context = service_context)
    index = VectorStoreIndex.from_documents([Document(text=documents[id]["text"])], callback_manager=callback_manager)
    
    logging.info(f"Loaded document with id {id}. Token count: {token_counter.total_embedding_token_count}")

def load_documents():
    return [
        {
            "id": "1",
            "text": "George Washington was the first President of the United States."
        },
        {
            "id": "2",
            "text": "He was the Commander-in-Chief of the Continental Army during the American Revolutionary War."
        },
        {
            "id": "3",
            "text": "He was one of the Founding Fathers of the United States."
        }
    ]

def _setup_service_context():
    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
    return service_context, token_counter

def _setup_token_counter():
    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])
    return token_counter, callback_manager