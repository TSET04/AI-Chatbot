import os
import mysql.connector
from dotenv import load_dotenv
from mistralai import Mistral
from langsmith import traceable


load_dotenv()

# Setting up database
connection = mysql.connector.connect(
    host = os.getenv("HOST"),
    user = os.getenv("USER"),
    password = os.getenv("PASSWORD"),
    database = os.getenv("DATABASE")
)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

model = "mistral-large-latest"
client = Mistral(api_key=MISTRAL_API_KEY)


prompt = [
    """
    You are an expert in converting English questions to SQL qyuery. The SQL database
    has the name Hospital_database with table name sample_data that has the following columns -
    ID, NAME and AGE 
    
    For example, 
    Example 1 - How many people are present, the SQL command will be 
    something like this- Select count(*) from sample_data;
    Example 2 - What is the age of Shiv ? the SQL
    command will be - Select age from sample_data where name = "Shiv";
    also the sql code should not have ``` in the beginning or end and sql word in output.
    """
]

@traceable
def call_model(user_message):
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "system",
                "content": prompt[0]
            },
            {
                "role": "user",
                "content": user_message,
            }
        ]
    )

    return chat_response.choices[0].message.content


def get_results_from_sql(query):
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    for d in data:
        print(d[0])

def text_to_sql_ai():
    t = 0
    while t<1:
        print("Ask me anything - ")
        user_msg = input()
        sql_query = call_model(user_msg)
        get_results_from_sql(sql_query)
        t+=1

text_to_sql_ai()
connection.close()