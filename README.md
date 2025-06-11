# SQL Agents with Multiple LLM Providers

A flexible system for querying databases using natural language through various LLM providers.

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Alternatively, use conda
# conda activate sql_tool_calling
```

## Configuration

All configurations are stored in the `conf` directory:

### Provider Configuration

Edit `conf/providers.json` to configure different LLM providers:

```json
{
  "openai": {
    "default_model": "gpt-4o-mini",
    "router_model": "gpt-4o-mini",
    "api_key_env": "OPENAI_API_KEY",
    "base_url": null
  },
  "ollama": {
    "default_model": "qwen2.5-coder:32b", //llama3.3
    "router_model": "qwen2.5:3b", //llama3.3
    "api_key": "ollama",
    "base_url": "http://localhost:11434/v1"
  },
  "gemini": {
    "default_model": "gemini-2.5-pro-exp-03-25",
    "router_model": "gemini-2.5-pro-exp-03-25",
    "api_key_env": "GEMINI_API_KEY",
    "base_url": null
  }
}
```

You can also add comments for alternative models with a separate section in the config:

```json
"ollama_models_comments": {
  "_comment": "qwen2.5-coder:32b, llama3.3, mistral-small3.1, etc."
},
"gemini_models_comments": {
  "_comment": "gemini-2.5-pro-exp-03-25, gemini-2.5-flash-preview-04-17, gemini-2.0-flash"
}
```

### Database Configuration

Edit `conf/database.json` to configure your database connection:

```json
{
  "postgres": {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "XXX",
    "port": "5432"
  }
}
```

### API Keys

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Ollama Setup

For Ollama, ensure the Ollama server is running locally before using:
```bash
# Install models you want to use
ollama pull qwen2.5:3b
ollama pull qwen2.5-coder:32b

# Or other models like DeepSeek
ollama pull deepseek-coder:r1

# Check available models
ollama list
```

## Usage

### Command Line Interface

Run the application with your preferred LLM provider:

```bash
# Use the unified script with provider selection
python run.py --provider openai  # Use OpenAI (default)
python run.py --provider ollama  # Use Ollama

# Or use the legacy scripts
python run_openai.py
python run_ollama.py
```

### Streamlit Web Interface

For a user-friendly web interface, run:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your browser. From the interface, you can:
- Select your preferred LLM provider (OpenAI or Ollama)
- Enter natural language queries for your database
- View responses and query results in a chat-like interface

## Semantic Database Documentation

When connecting to a new database, you should update the semantic model to provide appropriate context to the LLM agents. This is vital for accurate query generation.

### Update for New Databases

1. Edit `conf/semantic_model.json` to accurately describe your tables and relationships:
   - Update table names, descriptions, and use cases
   - Update the relationships between tables
   - Make sure all tables in your database are represented

2. Examples in the semantic model should follow this format:
   ```json
   {
     "tables": [
       {
         "table_name": "users",
         "table_description": "Contains user information",
         "Use Case": "Use this table to get user data"
       }
     ],
     "relationships": [
       {
         "from_table": "orders",
         "to_table": "users",
         "relationship": "Many orders belong to one user"
       }
     ]
   }
   ```

3. The system will automatically fetch column information from your database's information schema.

## Output

All queries and responses are saved to the `output` directory:
- `table_structure.md` and `table_structure.json`: Database schema information
- `query_TIMESTAMP.md`: Individual query results with token usage statistics