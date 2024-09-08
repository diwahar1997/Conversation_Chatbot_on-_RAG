
import os
import logging
import time
import uuid
from dotenv import load_dotenv
from flask import Flask, jsonify, request, Response, render_template
from flask_cors import CORS
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from prompt import prompt_template, contextualize_system_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})


class Chatbot:
    def __init__(self, vectors_database_location, model_name):
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        self.vectors_database_location = vectors_database_location
        self.embeddings = HuggingFaceEmbeddings(model_name=model)
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name,max_retries=2,streaming=True,max_tokens=2000,n=1)
        self.prompt = prompt_template()
        self.Chatbot_session_details='Chatbot_session_details'
        self.Chatbot_user_data_db_name='Chatbot_user_data_db_name'
        self.Chatbot_session_deleted_details='Chatbot_session_deleted_details'
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.session_info = dict()
        self.session_id_list=list() 
        
        
    def store_session_details(self):
        logger.info("Adding session details to MongoDB.")
        client = MongoClient("mongodb://localhost:27017/")
        db = client[self.Chatbot_user_data_db_name][self.Chatbot_session_details]
        logger.info("Storing all session data to MongoDB.")
        try:
            for session_id, chat_history in self.session_info.items():
                session_data = {
                    'session_id': session_id,
                    'chat_history': str(dict(chat_history))
                }
                db.update_one(
                        {'session_id': session_id},
                        {'$set': session_data},
                        upsert=True
                    )
            logger.info("All session data stored successfully.")
        except Exception as e:
            logger.error(f"Error storing session data: {e}")
    
    
    def store_session_details_sync(self):
        self.executor.submit(self.store_session_details)
                 
    
    def load_vector_database(self):
        vectors = FAISS.load_local(
            self.vectors_database_location, self.embeddings, allow_dangerous_deserialization=True)
        return vectors


    def initialize_chain(self, vectors_database):
        retriever = vectors_database.as_retriever()
        return retriever

    def Create_retrieval_chain(self, history_aware_retriever, chain):
        retrieval_chain = create_retrieval_chain(history_aware_retriever,chain)
        return retrieval_chain

    def get_response(self, conversational_retrieval_chain, user_input,session_id):
        response = conversational_retrieval_chain.invoke({"input": user_input},
        config={
        "configurable": {"session_id": session_id}
        })
        return response

    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.session_info:
                self.session_info[session_id] = ChatMessageHistory()
            return self.session_info[session_id]
        

    def handle_user_input(self, user_input,session_id):
        chain = create_stuff_documents_chain(self.llm, self.prompt)
        vectors_database = self.load_vector_database()
        retriever = self.initialize_chain(vectors_database)
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_system_prompt())
        retrieval_chain = self.Create_retrieval_chain(history_aware_retriever,chain)
        conversational_retrieval_chain = RunnableWithMessageHistory(
        retrieval_chain,
        self.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )
        
        def generate_response():
            response = self.get_response(conversational_retrieval_chain, user_input, session_id)
            self.store_session_details_sync()
            for chunk in response["answer"]:
                yield chunk
                time.sleep(0.05)
        return Response(generate_response(),headers={"session_id":session_id})


path = "trail_db"
name = "llama3-8b-8192"
model = 'all-MiniLM-L6-v2'
chat_bot = Chatbot(path, name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/session_id/<id>", methods=["DELETE"])
def delete_session_id(id):
    logger.info("Storing all deleted session data in MongoDB")
    client = MongoClient("mongodb://localhost:27017/")
    collection = client[chat_bot.Chatbot_user_data_db_name][chat_bot.Chatbot_session_deleted_details]
    session_conversation=str(dict(chat_bot.session_info[id]))
    insert_result = collection.insert_one({id: session_conversation})
    del chat_bot.session_info[id]
    logger.info(f"Inserted document ID: {insert_result.inserted_id}")
    return jsonify({"response": "Session cleared"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.get_json().get("message")
    session_id = request.get_json().get("session_id","")
    if not session_id:
        session_id=str(uuid.uuid4())
        
    app.logger.info(f'User Input : {user_input}')
    return chat_bot.handle_user_input(user_input,session_id)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
