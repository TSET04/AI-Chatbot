import os, time
import numpy as np
import faiss
import mysql.connector
from typing import Literal
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

# Setting up database
connection = mysql.connector.connect(
    host = os.getenv("HOST"),
    user = os.getenv("USER"),
    password = os.getenv("PASSWORD"),
    database = os.getenv("DATABASE")
)

client = Mistral(api_key=MISTRAL_API_KEY)

# Hyperparameters
chunk_size = 1000
chunk_overlap = 100
batch_size = 5
llm_model = "mistral-large-latest"
embed_model = "mistral-embed"

# Defining the Mistral Model
Model = ChatMistralAI(model=llm_model, temperature=0)

# Setting up Caching system
set_llm_cache(SQLiteCache(database_path="./cache/query_cache.db"))

# Loading the text file
loader = TextLoader("Healthcare_data.txt")
documents = loader.load()

# Splitting the text file
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
splits = text_splitter.split_documents(documents)

class MistralEmbedding(Embeddings):
    def embed_documents(self, texts):
        batch_size = 3
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(model=embed_model, inputs=batch)
            embeddings.extend([data.embedding for data in response.data])
            time.sleep(2)  # To avoid rate limits
        return embeddings

    def embed_query(self, text):
        time.sleep(1)
        if isinstance(text, dict):
            text = text.get("query", "")
        response = client.embeddings.create(model=embed_model, inputs=[text])
        return response.data[0].embedding

mistral_embedding = MistralEmbedding()

# FAISS VectorStore
vector_dim = 1024  # Adjust based on your embedding size
faiss_index = faiss.IndexFlatL2(vector_dim)

document_embeddings = mistral_embedding.embed_documents([doc.page_content for doc in splits])
faiss_index.add(np.array(document_embeddings, dtype=np.float32))

# Graph State Model
class QueryState(BaseModel):
    query: str
    datasource: str = ""
    response: str = ""

# Define LangGraph
workflow = StateGraph(QueryState)

# Implementing LLM based routing agent for graph's routing node
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["SQL Database", "Factual Database", "Irrelevant"] = Field(
        ...,
        description="Given a user question choose which database would be most relevant for answering their question",
    )


def route_query_node(state: QueryState):
    structured_llm = Model.with_structured_output(RouteQuery)

    # Routing Prompt 
    system = """You are an expert at routing user questions. There are 2 databases: 
    - **SQL (Hospital_database.sample_data)**: Stores structured patient data (ID, NAME, WEIGHT, AGE).
    - **Factual (General Information DB)**: Contains general knowledge.

    If the user question is irrelevant (not related to hospitals, healthcare, or health related factual knowledge), return 'irrelevant'.
    If the question contains gibberish, nonsense, or random text, also return 'irrelevant'.
    
    Rules:
    1. If the question is relevant to patient details (ID, NAME, WEIGHT, AGE), route it to 'SQL Database'.
    2. If the question is about general knowledge, route it to 'Factual Database'.
    3. If the question is unrelated or cannot be answered using the given databases, return 'Irrelevant'.
    4. If you are unsure, do NOT assume an answer. Return 'Irrelevant' instead.
    5. If the question contains gibberish, nonsense, personal question/remarks or random text, also return 'Irrelevant'.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    router = prompt | structured_llm
    category = router.invoke({"question": state.query})

    state.datasource = category
    return state.datasource

def irrelevant_node(state: QueryState):
    state.response = "I'm sorry, but I can only assist with healthcare-related or hospital based questions."
    return {"response": state.response}

def rag_node(state: QueryState):
    # Retrieving the docs
    time.sleep(1)
    query_embedding = np.array(mistral_embedding.embed_query(state.query), dtype=np.float32).reshape(1, -1)
    _, indices = faiss_index.search(query_embedding, k=3)
    retrieved_docs = [splits[i].page_content for i in indices[0]]
    context = "\n\n".join(retrieved_docs)

    factual_template = f"""
    Answer the following question based on this context:
    {context}
    Question: {state.query}
    """
    
    time.sleep(1)
    response = Model.invoke(factual_template)
    if isinstance(response, dict) and "content" in response:
        state.response = response["content"]
    elif hasattr(response, "content"):
        state.response = response.content
    else:
        state.response = str(response)

    return {"response": state.response}

def sql_node(state: QueryState):
    database_template = """
        You are an expert in converting English questions to SQL query. The SQL database
        has the name Hospital_database with table name sample_data that has the following columns -
        ID, NAME, WEIGHT and AGE 
        
        For example, 
        Example 1 - How many people are present, the SQL command will be 
        something like this- Select count(*) from sample_data;
        Example 2 - What is the age of Shiv ? the SQL
        command will be - Select age from sample_data where name = "Shiv";
        also the sql code should not have ``` in the beginning or end and sql word in output.
    """

    sql_prompt = database_template + f"\n\nQuestion: {state.query}\nSQL Query:"
    time.sleep(1)
    sql_query = Model.invoke(sql_prompt)
    cursor = connection.cursor()
    cursor.execute(sql_query.content)
    data = cursor.fetchall()
    state.response = ' '.join(str(d[0]) for d in data) if data else "No Data Found"
    return {"response":state.response}

# Adding nodes to the graph
workflow.add_node("route_query_node", route_query_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("sql_node", sql_node)
workflow.add_node("irrelevant_node", irrelevant_node)

# Adding edges to the graph
workflow.add_conditional_edges(
    "route_query_node",
    lambda s: "rag_node" if s.datasource == "Factual Database" else ("sql_node" if s.datasource=="SQL Database" else "irrelevant_node")
)
workflow.add_edge("rag_node", END)
workflow.add_edge("sql_node", END)
workflow.add_edge("irrelevant_node", END)

workflow.set_entry_point("route_query_node")
app = workflow.compile()

while True:
    user = input("\nUser: ")
    if user in ["quit", "q", "exit"]:
        connection.close()
        print("AI: Goodbye user. Have a nice day")
        break

    result = app.invoke(QueryState(query=user))
    print("AI: ", result.get('response', ""))