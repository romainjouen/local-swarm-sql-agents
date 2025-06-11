from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm
import json
import os
import psycopg2
from datetime import datetime
import re
import tiktoken

load_dotenv()

# Token counter class
class TokenCounter:
    def __init__(self, provider="openai", model_name=None):
        self.input_tokens = 0
        self.output_tokens = 0
        self.provider = provider
        self.model_name = model_name
        
    def add_input_tokens(self, count):
        self.input_tokens += count
        
    def add_output_tokens(self, count):
        self.output_tokens += count
        
    def get_total_tokens(self):
        return self.input_tokens + self.output_tokens
    
    def get_counts(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.get_total_tokens()
        }
    
    def count_tokens(self, text):
        """Count tokens for a given text string using tiktoken"""
        try:
            if self.provider == "openai":
                try:
                    # Try model-specific encoding for OpenAI first
                    if self.model_name:
                        encoding = tiktoken.encoding_for_model(self.model_name)
                    else:
                        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                    return len(encoding.encode(text))
                except:
                    # Fallback to general purpose encoding
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
            else: # ollama or other provider
                # For Ollama models, use a general purpose encoding
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
        except:
            # Last resort fallback - estimate based on characters
            return len(text) // 4

def create_token_counter(provider="openai", model_name=None):
    """Factory function to create a token counter for the specified provider"""
    return TokenCounter(provider=provider, model_name=model_name)

def get_database_connection():
    """Create a PostgreSQL database connection using config from conf/config.json"""
    try:
        # Load database configuration
        with open("conf/config.json", "r") as f:
            db_config = json.load(f)
        
        # Extract PostgreSQL configuration
        pg_config = db_config.get("postgres", {})
        
        # Create connection
        conn = psycopg2.connect(
            host=pg_config.get("host", "localhost"),
            dbname=pg_config.get("dbname", "postgres"),
            user=pg_config.get("user", "postgres"),
            password=pg_config.get("password", "admin"),
            port=pg_config.get("port", "5432")
        )
        
        return conn
    except Exception as e:
        print(f"Error creating database connection: {e}")
        raise

def get_model_config(provider="openai"):
    """Get model configuration for a specific provider from conf/config.json"""
    try:
        with open("conf/config.json", "r") as f:
            config = json.load(f)
        
        # Extract model configuration for the specified provider
        model_config = config.get("models", {}).get(provider, {})
        
        if not model_config:
            print(f"Warning: No model configuration found for provider '{provider}'. Using defaults.")
            # Return default values if provider configuration is missing
            return {
                "default": os.getenv('LLM_MODEL', 'gpt-4o-mini' if provider == "openai" else "llama3.3"),
                "router": os.getenv('ROUTER_MODEL', 'gpt-4o-mini' if provider == "openai" else "llama3.3")
            }
        
        return model_config
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        # Fall back to environment variables
        return {
            "default": os.getenv('LLM_MODEL', 'gpt-4o-mini' if provider == "openai" else "llama3.3"),
            "router": os.getenv('ROUTER_MODEL', 'gpt-4o-mini' if provider == "openai" else "llama3.3")
        }

def get_semantic_model():
    """Load the semantic model from conf/semantic_model.json"""
    try:
        with open("conf/semantic_model.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading semantic model: {e}")
        raise

def create_sql_executor(conn, token_counter):
    """Factory function to create a SQL executor function"""
    cursor = conn.cursor()
    
    def run_sql_select_statement(sql_statement):
        """Executes a SQL SELECT statement and returns the results of running the SELECT."""
        print(f"Executing SQL statement: {sql_statement}")
        try:
            # Count tokens for the SQL statement
            token_counter.add_input_tokens(token_counter.count_tokens(sql_statement))
            
            cursor.execute(sql_statement)
            records = cursor.fetchall()

            if not records:
                return "No results found."
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Calculate column widths
            col_widths = [len(name) for name in column_names]
            for row in records:
                for i, value in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(value)))
            
            # Format the results
            result_str = ""
            
            # Add header
            header = " | ".join(name.ljust(width) for name, width in zip(column_names, col_widths))
            result_str += header + "\n"
            result_str += "-" * len(header) + "\n"
            
            # Add rows
            for row in records:
                row_str = " | ".join(str(value).ljust(width) for value, width in zip(row, col_widths))
                result_str += row_str + "\n"
            
            # Count tokens for the result
            token_counter.add_output_tokens(token_counter.count_tokens(result_str))
            
            return result_str
        
        except psycopg2.Error as e:
            error_msg = f"PostgreSQL error: {str(e)}"
            token_counter.add_output_tokens(token_counter.count_tokens(error_msg))
            return error_msg
        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            token_counter.add_output_tokens(token_counter.count_tokens(error_msg))
            return error_msg
    
    return run_sql_select_statement

def list_columns(run_sql_select_statement):
    """Get a list of all columns from all tables."""
    list_of_columns = run_sql_select_statement("""
        SELECT 
            table_name, 
            column_name 
        FROM 
            information_schema.columns 
        WHERE 
            table_schema = 'public' 
        ORDER BY 
            table_name, 
            ordinal_position
    """)
    
    # Ensure output directory exists
    os.makedirs("output/tables_structure", exist_ok=True)
    
    with open("output/tables_structure/table_structure.md", "w") as f:
        f.write(list_of_columns)
    with open("output/tables_structure/table_structure.json", "w") as f:
        json.dump(list_of_columns, f, indent=2)  
  
    return list_of_columns

def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")

def extract_sql_query(messages):
    for message in messages:
        if "tool_calls" in message and message["tool_calls"] is not None:
            for tool_call in message["tool_calls"]:
                if tool_call["function"]["name"] == "run_sql_select_statement":
                    args = json.loads(tool_call["function"]["arguments"])
                    # Handle both 'sql_statement' and 'query' parameter names
                    return args.get('sql_statement', args.get('query', ''))
    return ""

def save_to_markdown(user_query, result, sql_query, token_counter, begin_time=None, end_time=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/results/query_{timestamp}.md"
    
    # Get token counts
    token_counts = token_counter.get_counts()
    
    # Format timestamps for display
    begin_timestamp = begin_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_timestamp = end_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""## User query
{user_query}

## Provider
- Provider: {token_counter.provider}
- Model: {token_counter.model_name}

## Processing Time
- Begin: {begin_timestamp}
- End: {end_timestamp}

## Result
{result}

## SQL
{sql_query}

## Token Usage
- Input tokens: {token_counts['input_tokens']}
- Output tokens: {token_counts['output_tokens']}
- Total tokens: {token_counts['total_tokens']}
"""
    
    with open(filename, "w") as f:
        f.write(markdown_content)
    print(f"\033[92mResults saved to {filename}\033[0m")

def run_demo_loop(
    client, 
    starting_agent, 
    token_counter,
    provider_name="AI",
    context_variables=None, 
    stream=False, 
    debug=False
) -> None:
    swarm = Swarm(client=client)
    print(f"Starting {provider_name} Swarm CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        # Record begin timestamp
        begin_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        response = swarm.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        # Record end timestamp
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)
            
            # Extract the last assistant message as the result
            last_assistant_message = next(
                (msg for msg in reversed(response.messages) if msg["role"] == "assistant"),
                None
            )
            
            if last_assistant_message:
                result = last_assistant_message.get("content", "")
                sql_query = extract_sql_query(response.messages)
                save_to_markdown(user_input, result, sql_query, token_counter, begin_timestamp, end_timestamp)

        messages.extend(response.messages)
        agent = response.agent 