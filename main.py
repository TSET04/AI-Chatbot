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
from conversation_memory import ConversationMemory
from System_Prompts import memory_prommpt, route_prompt, rag_node_prompt, sql_node_prompt, summarizer_prompt, sql_count_prompt, format_response

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

# Setting up Memory system
memory = ConversationMemory()

# Loading the text file
loader = TextLoader("Healthcare_data.txt")
documents = loader.load()

# Splitting the text file
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
splits = text_splitter.split_documents(documents)

# Custom Embedding class
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
    sql_subqueries: list[str] = []

# Class for histor storage
class SummaryAndDict(BaseModel):
    summary: str
    key_value_pairs: dict

# Define LangGraph
workflow = StateGraph(QueryState)

def memory_node(state: QueryState):
    query = state.query
    history = memory.get_history()    
    response = Model.invoke(memory_prommpt.format(query=query, history=history)).content

    if response.startswith("UNKNOWN"):
        return state
    
    state.response = response
    return {"response": response}

# Implementing LLM based routing agent for graph's routing node
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["SQL Database", "Factual Database", "Irrelevant"] = Field(
        ...,
        description="Given a user question choose which database would be most relevant for answering their question",
    )

# Route query node to choose for SQL/RAG/Irrelevant datasources
def route_query_node(state: QueryState):
    time.sleep(1)
    structured_llm = Model.with_structured_output(RouteQuery)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", route_prompt),
            ("human", "{question}"),
        ]
    )
    router = prompt | structured_llm
    category = router.invoke({"question": state.query})

    state.datasource = category
    return state.datasource

# Irrelevant node datasource
def irrelevant_node(state: QueryState):
    state.response = "I'm sorry, but I can only assist with healthcare-related or hospital based questions."
    return {"response": state.response}

# Rag node for factual database
def rag_node(state: QueryState):
    # Retrieving the docs
    time.sleep(1)
    query_embedding = np.array(mistral_embedding.embed_query(state.query), dtype=np.float32).reshape(1, -1)
    _, indices = faiss_index.search(query_embedding, k=3)
    retrieved_docs = [splits[i].page_content for i in indices[0]]
    context = "\n\n".join(retrieved_docs)

    time.sleep(1)
    response = Model.invoke(rag_node_prompt.format(context=context, question=state.query))
    if isinstance(response, dict) and "content" in response:
        state.response = response["content"]
    elif hasattr(response, "content"):
        state.response = response.content
    else:
        state.response = str(response)

    return {"response": state.response}

# Function for extracting multiple sql questions out of a single one
def sql_query_count(state: QueryState):
    time.sleep(1)
    query = state.query

    response = Model.invoke(sql_count_prompt.format(question=query)).content
    if response.startswith("MULTIPLE"):
        decomposed = [q.strip() for q in response.replace("MULTIPLE:", "").split("\n") if q.strip()]
        clean_queries = []
        for q in decomposed:
            if q[0].isdigit() and ". " in q:
                clean_queries.append(q.split(". ", 1)[1])
            else:
                clean_queries.append(q)
                
        state.sql_subqueries = clean_queries
    else:
        state.sql_subqueries = []

    return state

# Function to fetch from sql db
def fetch_sql_results(question):
    try:
        sql_query = Model.invoke(sql_node_prompt.format(question=question))
        cursor = connection.cursor()
        cursor.execute(sql_query.content)
        data = cursor.fetchall()
        result = ' '.join(str(d[0]) for d in data) if data else "No Data Found"
        return result
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# Sql node to execute sql based questions
def sql_node(state: QueryState):
    time.sleep(1)
    if state.sql_subqueries:
        results = []
        for sub_query in state.sql_subqueries:
            sub_result = fetch_sql_results(sub_query)
            results.append(f"For '{sub_query}': {sub_result}")
    
        state.response = "\n\n".join(results)
    else:
        # Original single-query logic
        data = fetch_sql_results(state.query)
        state.response = data if data else "No Data Found"

    return {"response": state.response}

# Function to return a human - friendly response from model
def format_response_node(state: QueryState):
    time.sleep(1)
    query = state.query
    response = state.response

    result = Model.invoke(format_response.format(question=query, response=response)).content
    state.response = result
    return {"response": state.response}


# Function to store the summary of the question - answer pairs
def my_summarizer(state: QueryState):
    time.sleep(1)
    query = state.query
    response = state.response
    structured_llm = Model.with_structured_output(SummaryAndDict)
    result = structured_llm.invoke(summarizer_prompt.format(question=query, response=response))
    summary = result.summary
    key_value_pairs = result.key_value_pairs

    memory.add(query, summary, key_value_pairs)
    return state

# Adding nodes to the graph
workflow.add_node("memory_node", memory_node)
workflow.add_node("route_query_node", route_query_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("sql_query_count", sql_query_count)
workflow.add_node("sql_node", sql_node)
workflow.add_node("irrelevant_node", irrelevant_node)
workflow.add_node("my_summarizer", my_summarizer)
workflow.add_node("format_response_node", format_response_node)

# Adding edges to the graph
workflow.add_conditional_edges(
    "route_query_node",
    lambda s: ("sql_query_count" if s.datasource == "SQL Database" else 
              ("rag_node" if s.datasource == "Factual Database" else "irrelevant_node"))
)

workflow.add_conditional_edges(
    "memory_node",
    lambda s: END if s.response else "route_query_node"
)

workflow.add_edge("sql_query_count", "sql_node")
workflow.add_edge("rag_node", "my_summarizer")
workflow.add_edge("sql_node", "my_summarizer")
workflow.add_edge("my_summarizer", "format_response_node")
workflow.add_edge("irrelevant_node", END)
workflow.add_edge("format_response_node", END)

workflow.set_entry_point("memory_node")
app = workflow.compile()

while True:
    user = input("\nUser: ")
    if user in ["quit", "q", "exit"]:
        connection.close()
        print("AI: Goodbye user. Have a nice day")
        break

    result = app.invoke(QueryState(query=user))
    print("AI: ", result.get('response', ""))