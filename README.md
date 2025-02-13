# Chat2SQLDB
GenAI Open AI based Chat with SQL Database

Create environment with python > 3.11

pip install -r requirements.txt

cd Csv_Sqlite3_OpenAI

set OpenAI API key using below script 
#Python 
# import os
# os.environ["OPENAI_API_KEY"]="<PASTE OPENAPI API KEY HERE>"
# print(os.getenv("OPENAI_API_KEY"))

#cmd line (Linux/MacOS)
export OPENAI_API_KEY="<PASTE OPENAPI API KEY HERE>"
echo $OPENAI_API_KEY

#Cmdline (Windows)
set OPENAI_API_KEY=<PASTE OPENAPI API KEY HERE>
echo %OPENAI_API_KEY%

python llm2sqldb.py