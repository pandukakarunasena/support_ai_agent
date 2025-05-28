# async_mcp_context.py
import contextvars

conversation_id_var = contextvars.ContextVar("conversation_id")

def set_conversation_id(cid: str):
    conversation_id_var.set(cid)

def get_conversation_id() -> str:
    return conversation_id_var.get(None)