from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

def prompt_template():
    system_prompt = (
    "You are an assistant for question-answering tasks"
    "Use the following pieces of retrieved context to answer the question"
    "If you don't know the answer, say that you <p>My knowledge is limited to Fireflink-related topics. Please ask me a question about Fireflink.</p>"
    "keep the answer concise."
    "using HTML tags for formatting."
    "\n\n"
    "{context}"
    )
    return ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        
    ]
    )

    
def contextualize_system_prompt():
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "with the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    return  ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
    )