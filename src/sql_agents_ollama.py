from dotenv import load_dotenv
from swarm import Agent
import json
import os
from datetime import datetime
import re
from .sql_agents_common import (
    create_token_counter, 
    get_database_connection, 
    get_semantic_model, 
    create_sql_executor,
    list_columns,
    get_model_config
)

load_dotenv()

# Get model configuration for Ollama
model_config = get_model_config("ollama")
default_model = model_config.get("default")
router_model = model_config.get("router", default_model)

# Create token counter instance
token_counter = create_token_counter(provider="ollama", model_name=default_model)

# Get PostgreSQL connection
conn = get_database_connection()

# Create SQL executor function
run_sql_select_statement = create_sql_executor(conn, token_counter)

# Get table schemas from semantic model
table_schemas = get_semantic_model()

def get_sql_router_agent_instructions():
    return """You are an orchestrator of different SQL data experts and it is your job to
    determine which of the agent is best suited to handle the user's request, 
    and transfer the conversation to that agent."""

def get_sql_agent_instructions(list_of_columns):
    # Format tables information
    tables_info = ""
    for table in table_schemas["tables"]:
        tables_info += f"\n## {table['table_name']}\n"
        tables_info += f"Description: {table['table_description']}\n"
        tables_info += f"Use Case: {table['Use Case']}\n"
    
    # Format relationships information
    relationships_info = "\n## Relationships\n"
    for rel in table_schemas["relationships"]:
        relationships_info += f"- {rel['from_table']} → {rel['to_table']}: {rel['relationship']}\n"
    
    return f"""You are a SQL expert who takes in a request from a user for information
    they want to retrieve from the DB, creates a SELECT statement to retrieve the
    necessary information, and then invoke the function to run the query and
    get the results back to then report to the user the information they wanted to know.
    
    Here are the tables and their descriptions for the DB you can query:
    {tables_info}
    
    Here are the relationships between the tables:
    {relationships_info}
    
    Here are all the columns available in the database:
    {list_of_columns}

    Write all of your SQL SELECT statements to work 100% with these schemas and nothing else.
    You are always willing to create and execute the SQL statements to answer the user's question.
    """

# Get list of columns for instructions
column_list = list_columns(run_sql_select_statement)

sql_router_agent = Agent(
    name="Router Agent",
    instructions=get_sql_router_agent_instructions(),
    model=router_model
)
data_query_agent = Agent(
    name="Data Query Agent",
    instructions=get_sql_agent_instructions(column_list) + "\n\nHelp the user with data in the database. Be super enthusiastic about.",
    functions=[run_sql_select_statement],
    model=default_model
)
user_agent = Agent(
    name="User Agent",
    instructions=get_sql_agent_instructions(column_list) + "\n\nHelp the user with data related to users.",
    functions=[run_sql_select_statement],
    model=default_model
)
analytics_agent = Agent(
    name="Analytics Agent",
    instructions=get_sql_agent_instructions(column_list) + "\n\nHelp the user gain insights from the data with analytics. Be super accurate in reporting numbers and citing sources.",
    functions=[run_sql_select_statement],
    model=default_model
)

def transfer_back_to_router_agent(*args, **kwargs):
    """Call this function if a user is asking about data that is not handled by the current agent."""
    return sql_router_agent

def transfer_to_data_query_agent(*args, **kwargs):
    return data_query_agent

def transfer_to_user_agent(*args, **kwargs):
    return user_agent

def transfer_to_analytics_agent(*args, **kwargs):
    return analytics_agent

sql_router_agent.functions = [transfer_to_data_query_agent, transfer_to_user_agent, transfer_to_analytics_agent]
data_query_agent.functions.append(transfer_back_to_router_agent)
user_agent.functions.append(transfer_back_to_router_agent)
analytics_agent.functions.append(transfer_back_to_router_agent)