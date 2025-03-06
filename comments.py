# Implementing RAG-Fusion (Query-translation - 1)

def reciprocal_rank_fusion(documents: list[list], k=20):
    fused_scores = {}
    for docs in documents:
        for rank, doc in enumerate(docs):
            doc = str(doc)
            fused_scores[doc] = fused_scores.get(doc, 0) + 1/(rank+k)
    
    reranked_results = [(doc, score) for doc, score in sorted(fused_scores.items(), key = lambda x: x[1], reverse=True)]
    return reranked_results

reranked_docs = reciprocal_rank_fusion(docs)
top_k_docs = [doc[0] for doc in reranked_docs[:5]]  # Select top 5 documents
context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(top_k_docs)]) # Better structured docs

# Rule-Based Classification Keywords
db_keywords = {"age", "name", "id", "Weight"}

# Classification based on db keywords
def rule_based_classification(input_data):
    question = input_data.get("question", "")
    if any(keyword in question.lower() for keyword in db_keywords):
        return "database"
    return "factual"

# Load pre-trained ML-based intent classifier
with open("intent_classifier.pkl", "rb") as f:
    intent_classifier = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def ml_based_classification(question):
    query_vector = vectorizer.transform([question])
    return intent_classifier.predict(query_vector)[0]

# Semantic Routing process
prmopt_templates = [factual_template, database_template]
prompt_embeddings = [mistral_embedding.embed_documents([template])[0] for template in prmopt_templates]

def semantic_routing(question):
    query_embedding = MistralEmbedding().embed_query(question)
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    return "database" if similarity.argmax() == 1 else "factual"

# Final Router combining all methods
def prompt_router(input_data):
    query = input_data.get("query", "")
    
    # Step 1: Rule-Based Classification
    rule_result = rule_based_classification(query)
    if rule_result:
        return {"query": query, "intent": rule_result}
    
    # Step 2: ML-Based Classification
    ml_result = ml_based_classification(query)
    if ml_result:
        return {"query": query, "intent": ml_result}
    
    # Step 3: Semantic Similarity Routing
    sem_result = semantic_routing(query)
    return {"query": query, "intent": sem_result}

# Recursive execution or retrive
def execute_or_retrieve(input_data, rerouted=False):
    if input_data["intent"] == "database":
        sql_query = generate_sql_query(input_data["query"])
        cursor = connection.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
        cursor.close()
        
        if not data and not rerouted:  # If no result and not already rerouted
            new_intent = "factual" if input_data["intent"] == "database" else "database"
            return execute_or_retrieve({"query": input_data["query"], "intent": new_intent}, rerouted=True)
        
        return " ".join(str(d[0]) for d in data) if data else "No relevant data found."

    else:
        response = ChatMistralAI(model="mistral-large-latest", temperature=0).invoke(
            factual_template.format(context=context, question=input_data["query"])
        )
        
        if ("context not found" in response.content.lower() or not context) and not rerouted:
            new_intent = "database"
            return execute_or_retrieve({"query": input_data["query"], "intent": new_intent}, rerouted=True)
        
        return response.content.strip()



# Function to add into memory
def save_to_memory(query, query_type, response):
    with open(MEMORY_FILE, "r") as f:
        history = json.load(f)
    
    history.append({"query": query, "query_type": query_type, "response": response})

    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# Function to call memory
def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return f"Error occurred {e}"
