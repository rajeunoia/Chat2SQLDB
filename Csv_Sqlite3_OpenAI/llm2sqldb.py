# %% [markdown]
# # Connecting to a SQL Database

# %% [markdown]
# ## Setup

# %%
import os
from IPython.display import Markdown, HTML, display
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
#from langchain_openai import AzureChatOpenAI
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import gradio as gr

# %% [markdown]
# ## Recover the original dataset

# %% [markdown]
# **Note**: To access the data locally, use the following code:
# 
# ```
# os.makedirs("data",exist_ok=True)
# !wget https://covidtracking.com/data/download/all-states-history.csv -P ./data/
# file_url = "./data/all-states-history.csv"
# df = pd.read_csv(file_url).fillna(value = 0)
# ```

# %%
from sqlalchemy import create_engine
import pandas as pd


df = pd.read_csv("./data/all-states-history.csv").fillna(value = 0)

# %% [markdown]
# ## Move the data to the SQL database

# %%
# Path to your SQLite database file
database_file_path = "./db/test.db"

# Create an engine to connect to the SQLite database
# SQLite only requires the path to the database file
engine = create_engine(f'sqlite:///{database_file_path}')
file_url = "./data/all-states-history.csv"
df = pd.read_csv(file_url).fillna(value = 0)
df.to_sql(
    'all_states_history',
    con=engine,
    if_exists='replace',
    index=False
)

# %% [markdown]
# ## Prepare the SQL prompt

# %%
MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.

## Tools:

"""

# %%
MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [death]
FROM covidtracking 
WHERE state = 'TX' AND date LIKE '2020%'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 people who died of covid in Texas in 2020.

Explanation:
I queried the `covidtracking` table for the `death` column where the state
is 'TX' and the date starts with '2020'. The query returned a list of tuples
with the number of deaths for each day in 2020. To answer the question,
I took the sum of all the deaths in the list, which is 27437.
I used the following query

```sql
SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
```
===> End of Example

"""

# %% [markdown]
# ## Call the OpenAI  Chat model and create the SQL agent

# %% [markdown]
# API Key needs to be set in the environment , 
# 
# import os
# os.environ["OPENAI_API_KEY"]="<PASTE OPENAPI API KEY HERE>"
# print(os.getenv("OPENAI_API_KEY"))


# %%
# Load the Ollama model
#model_name = "llama3.2:1b" #"wizardcoder:python"
#llm = Ollama(model=model_name)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=2,
    #api_key="<OPENAI API KEY>", #If environment based is not working
    #max_tokens=1000,
    # base_url="...",
    # organization="...",
)

# Connect to the local SQLite database
database_uri = "sqlite:///db/test.db"
db = SQLDatabase.from_uri(database_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# %%
QUESTION = """Get no of patients per year from FL and NY
Use the hospitalizedIncrease column
"""

agent_executor_SQL = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=llm,
    toolkit=toolkit,
    top_k=30,
    verbose=True
)

# %% [markdown]
# ## Invoke the SQL model

# %%
#agent_executor_SQL.invoke(QUESTION)

# %%


# Function to handle user queries
def query_database(user_query):
    try:
        result = agent_executor_SQL.run(user_query)
        return result
    except Exception as e:
        return str(e)

# Create Gradio interface
iface = gr.Interface(
    fn=query_database,
    inputs=gr.Textbox(lines=2, placeholder="Enter your SQL query in natural language..."),
    outputs="text",
    title="Chat2SQL Database Query",
    description="Ask questions about your SQLite database in plain English, and get SQL-based answers.",
)

# Launch the app
# Launch the UI and open browser automatically
iface.launch(share=False, inbrowser=True)

# %%



