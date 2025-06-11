import streamlit as st
import json
import os
import subprocess
import re
import pandas as pd
import glob
from datetime import datetime
import time
import sys

st.set_page_config(
    page_title="SQL Agent Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme to entire app
st.markdown("""
<style>
/* Dark theme for Streamlit */
.stApp {
    background-color: #121212;
    color: white;
}
.stTextInput, .stTextArea, .stSelectbox, .stMultiselect {
    color: white;
}
.stButton>button {
    background-color: #4c78a8;
    color: white;
}
.stSidebar {
    background-color: #1e1e1e;
    color: white;
}
.stSidebar .stMarkdown, .stSidebar .stSelectbox {
    color: white;
}
h1, h2, h3, h4, h5, h6, p {
    color: white !important;
}
.stInfo {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
.stError {
    background-color: #4e1e1e;
    color: #d4d4d4;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    body {
        color: #f8f8f2;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c78a8 !important;
        color: white !important;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .sql-highlight {
        background-color: #1e1e1e;
        color: #e6e6e6;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        line-height: 1.5;
        margin: 10px 0;
    }
    .stTabs {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
    }
    .token-counter {
        font-size: 0.8rem;
        color: #808080;
        margin-top: 5px;
    }
    .blue-text {
        color: #4c78a8;
    }
    .green-text {
        color: #59a14f;
    }
    .code-keyword {
        color: #569cd6;
        font-weight: bold;
    }
    .code-from {
        color: #569cd6;
        font-weight: bold;
    }
    .code-where {
        color: #569cd6;
        font-weight: bold;
    }
    .code-and {
        color: #d7ba7d;
        font-weight: bold;
    }
    .code-between {
        color: #d7ba7d;
        font-weight: bold;
    }
    .code-value {
        color: #ce9178;
    }
    .code-equal {
        color: #d4d4d4;
    }
    .code-table {
        color: #9cdcfe;
    }
    .bold-white {
        color: white;
        font-weight: bold;
    }
    .step-section {
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .user-id {
        font-weight: bold;
    }
    .result-highlight {
        font-size: 22px;
        font-weight: bold;
        margin: 20px 0;
    }
    .token-display {
        display: flex;
        justify-content: space-between;
        padding: 20px 0;
        border-top: 1px solid #d4d4d4;
        border-bottom: 1px solid #d4d4d4;
        margin: 20px 0;
    }
    .token-column {
        text-align: center;
    }
    .token-value {
        font-size: 36px;
        font-weight: bold;
    }
    .token-label {
        font-size: 18px;
        color: #777;
    }
    .sql-code {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        line-height: 1.6;
        margin: 10px 0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .sql-keyword { color: #569cd6; font-weight: normal; }
    .sql-alias { color: #569cd6; }
    .sql-function { color: #dcdcaa; }
    .sql-operator { color: #d7ba7d; }
    .sql-join { color: #569cd6; }
    .sql-table { color: white; }
    .sql-column { color: white; }
    .sql-string { color: #ce9178; }
    .sql-number { color: #b5cea8; }
    .sql-comment { color: #6a9955; }
    .sql-comma { color: #e6e6e6; }
    .sql-punc { color: #e6e6e6; }
</style>
""", unsafe_allow_html=True)

def highlight_sql(sql):
    """Add syntax highlighting to SQL query"""
    # Make a copy of the original SQL to avoid modifying the input directly
    highlighted_sql = sql
    
    # SQL keywords in blue - use word boundaries to avoid partial matches
    highlighted_sql = re.sub(r'\b(SELECT|FROM|WHERE|JOIN|ON|AS)\b', 
                            r'<span class="code-keyword">\1</span>', 
                            highlighted_sql, 
                            flags=re.IGNORECASE)
    
    # Operators and modifiers in orange
    highlighted_sql = re.sub(r'\b(AND|OR|NOT|IN|BETWEEN|LIKE|ILIKE)\b', 
                           r'<span class="code-and">\1</span>', 
                           highlighted_sql, 
                           flags=re.IGNORECASE)
    
    # COUNT, DISTINCT and other functions with special coloring
    highlighted_sql = re.sub(r'\b(COUNT|DISTINCT|SUM|AVG|MAX|MIN)\b', 
                           r'<span style="color: #61afef;">\1</span>', 
                           highlighted_sql, 
                           flags=re.IGNORECASE)
    
    # Table and column names followed by a dot
    highlighted_sql = re.sub(r'\b([a-zA-Z][a-zA-Z0-9_]*)\s*\.', 
                           r'<span class="code-table">\1</span>.', 
                           highlighted_sql)
    
    # Equal signs and operators - be careful with = to avoid conflicts with HTML attributes
    highlighted_sql = re.sub(r'(=|<>|<|>|\+|\-|\*|/)', 
                           r'<span class="code-equal">\1</span>', 
                           highlighted_sql)
    
    # Handle string literals in single quotes - this needs special care to avoid overlapping with other replacements
    def replace_string_literal(match):
        return f'<span style="color: #98c379;">{match.group(0)}</span>'
    
    # Use a more careful approach for string literals
    highlighted_sql = re.sub(r'\'[^\']*\'', replace_string_literal, highlighted_sql)
    
    # Handle percentage signs in string literals
    highlighted_sql = re.sub(r'<span style="color: #98c379;">\'%([^\']*?)%\'</span>', 
                           r'<span style="color: #98c379;">\'%\1%\'</span>', 
                           highlighted_sql)
    
    # Parentheses with subtle highlight
    highlighted_sql = re.sub(r'(\(|\))', 
                           r'<span style="color: #e6e6e6;">\1</span>', 
                           highlighted_sql)
    
    # Semicolons
    highlighted_sql = re.sub(r'(;)', 
                           r'<span style="color: #e6e6e6;">\1</span>', 
                           highlighted_sql)
    
    return highlighted_sql

def modify_run_py_for_non_interactive():
    """Create a temporary modified version of run.py that doesn't need interactive input"""
    # Make sure the directory exists
    os.makedirs("_temp", exist_ok=True)
    
    # Read the original run.py
    with open("run.py", "r") as f:
        content = f.read()
    
    # Modify the content to accept command line argument for query
    modified_content = content.replace(
        "def main():",
        """def main():
    # Check if direct query is provided as argument
    if len(sys.argv) > 2 and sys.argv[2].startswith("--query="):
        direct_query = sys.argv[2][8:]  # Remove the --query= prefix
    else:
        direct_query = None"""
    )
    
    # Replace the input() function with a non-interactive alternative
    modified_content = modified_content.replace(
        "user_input = input(\"\\033[90mUser\\033[0m: \")",
        """if direct_query:
            user_input = direct_query
            print(f"\\033[90mUser\\033[0m: {user_input}")
            # Exit loop after processing the direct query
            is_direct_query = True
        else:
            user_input = input("\\033[90mUser\\033[0m: ")
            is_direct_query = False"""
    )
    
    # Add a break after processing a direct query
    modified_content = modified_content.replace(
        "agent = response.agent",
        """agent = response.agent
        
        # If this was a direct query from command line, exit after one iteration
        if 'is_direct_query' in locals() and is_direct_query:
            break"""
    )
    
    # Add sys import and path setup to find modules
    modified_content = modified_content.replace(
        "import importlib",
        """import importlib
import sys
import os

# Add the parent directory to the Python path so we can find the src module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))"""
    )
    
    # Write the modified content to a temporary file
    with open("_temp/run_non_interactive.py", "w") as f:
        f.write(modified_content)
    
    return "_temp/run_non_interactive.py"

def run_query(provider, user_query):
    """Run the query using the specified provider"""
    import sys
    
    # Create output directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/results", exist_ok=True)
    
    # Get the list of existing result files before running
    before_files = set(glob.glob("output/results/query_*.md"))
    
    try:
        st.write(f"Running query with {provider} provider...")
        
        # Properly handle multi-line queries by joining with space
        user_query = ' '.join(user_query.strip().split('\n'))
        
        # Import necessary modules directly
        from dotenv import load_dotenv
        from openai import OpenAI
        import importlib
        
        # Load environment variables
        load_dotenv()
        
        # Dynamically import the appropriate modules based on provider - exact same approach as run.py
        provider_module = importlib.import_module(f"src.sql_agents_{provider}")
        
        # Get router agent and token counter from the imported module - exact same approach as run.py
        sql_router_agent = provider_module.sql_router_agent
        token_counter = provider_module.token_counter
        
        # Initialize the client - exact same approach as run.py
        if provider == 'openai':
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            provider_name = "OpenAI"
        else:  # ollama
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            provider_name = "Ollama"
        
        # Import common modules
        from src.sql_agents_common import run_demo_loop, save_to_markdown, extract_sql_query
        from swarm import Swarm
        from datetime import datetime
        
        # Create a non-interactive version of run_demo_loop for single queries
        def run_single_query(user_query):
            swarm = Swarm(client=client)
            messages = [{"role": "user", "content": user_query}]
            
            # Record begin timestamp
            begin_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Run the query - exactly like run.py/run_demo_loop without context_variables
            response = swarm.run(
                agent=sql_router_agent,
                messages=messages,
                stream=False,
                debug=False,
            )
            
            # Record end timestamp
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract the last assistant message as the result
            last_assistant_message = next(
                (msg for msg in reversed(response.messages) if msg["role"] == "assistant"),
                None
            )
            
            if last_assistant_message:
                result = last_assistant_message.get("content", "")
                sql_query = extract_sql_query(response.messages)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/results/query_{timestamp}.md"
                save_to_markdown(user_query, result, sql_query, token_counter, begin_timestamp, end_timestamp)
                return True, filename
            else:
                return False, "No result was generated."
        
        # Execute the query
        success, result_file = run_single_query(user_query)
        
        # Wait a bit for file writing to complete
        time.sleep(1)
        
        # Get the list of result files after running
        after_files = set(glob.glob("output/results/query_*.md"))
        
        # Find new files
        new_files = after_files - before_files
        
        if new_files:
            # Return the latest file
            latest_file = max(new_files, key=os.path.getmtime)
            return True, latest_file
        else:
            return False, "No result file was generated."
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, f"Error running query: {str(e)}\n\nDetails:\n{error_details}"

def read_latest_result(file_path=None):
    """Read the most recent result file from output/results or a specific file"""
    if file_path and os.path.exists(file_path):
        target_file = file_path
    else:
        results_dir = "output/results"
        if not os.path.exists(results_dir):
            return None
        
        files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) 
                if f.startswith("query_") and f.endswith(".md")]
        
        if not files:
            return None
        
        target_file = max(files, key=os.path.getmtime)
    
    with open(target_file, "r") as f:
        content = f.read()
    
    # Parse the markdown file
    sections = {}
    current_section = None
    current_content = []
    
    for line in content.split("\n"):
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = "\n".join(current_content)
            current_section = line[3:]
            current_content = []
        else:
            current_content.append(line)
    
    if current_section is not None:
        sections[current_section] = "\n".join(current_content)
    
    return sections

def parse_table_data(result_text):
    """Parse the result text to extract table data"""
    # Look for a markdown table in the text
    lines = result_text.strip().split("\n")
    
    # Find lines that look like a table header
    table_start = None
    for i, line in enumerate(lines):
        if "|" in line and "---" not in line:
            # Look for header candidate
            header_candidate = [col.strip() for col in line.split("|") if col.strip()]
            # Check if next line has separator
            if i+1 < len(lines) and all("-" in cell for cell in lines[i+1].split("|") if cell.strip()):
                table_start = i
                break
    
    if table_start is None:
        # Try to find a vertical bar table with common column names
        common_columns = ["id", "user_id", "interaction_id", "title", "article", "time", "date"]
        for i, line in enumerate(lines):
            cols = [col.strip().lower() for col in line.split("|") if col.strip()]
            if any(common in cols for common in common_columns) and "|" in line:
                table_start = i
                break
    
    if table_start is None:
        return None
    
    # Extract header and data
    header = [h.strip() for h in lines[table_start].split("|") if h.strip()]
    
    # Find where the separator row is
    separator_row = table_start + 1
    while separator_row < len(lines) and not all("-" in cell for cell in lines[separator_row].split("|") if cell.strip()):
        separator_row += 1
    
    if separator_row >= len(lines):
        separator_row = table_start + 1  # Just assume it's the next row
    
    data = []
    for i in range(separator_row + 1, len(lines)):
        if "|" not in lines[i] or lines[i].strip() == "":
            continue
        row_values = [val.strip() for val in lines[i].split("|") if val.strip()]
        if len(row_values) >= len(header):
            # Trim to match header length
            data.append(row_values[:len(header)])
    
    # If we didn't find any data rows, try a different approach
    if not data:
        for i in range(table_start + 1, len(lines)):
            if "|" not in lines[i] or lines[i].strip() == "":
                continue
            if "---" in lines[i]:
                continue
            row_values = [val.strip() for val in lines[i].split("|") if val.strip()]
            if len(row_values) > 0:
                # If row has fewer columns than header, pad with empty values
                while len(row_values) < len(header):
                    row_values.append("")
                # Trim to match header length
                data.append(row_values[:len(header)])
    
    # Convert to dataframe if we have data
    if data:
        df = pd.DataFrame(data, columns=header)
        return df
    
    return None

def display_sql(sql_query):
    """Display SQL query with proper syntax highlighting to match the screenshots exactly"""
    # Define SQL formatting styles to match the first screenshot
    sql_lines = []
    
    # Try to format the SQL to match the first screenshot
    if "SELECT" in sql_query and "FROM" in sql_query:
        try:
            # Convert to uppercase for keywords to help with formatting
            sql_upper = sql_query.upper()
            sql_parts = {
                "SELECT": sql_query[sql_upper.find("SELECT"):sql_upper.find("FROM")],
                "FROM": sql_query[sql_upper.find("FROM"):sql_upper.find("WHERE") if "WHERE" in sql_upper else len(sql_query)],
                "WHERE": sql_query[sql_upper.find("WHERE"):] if "WHERE" in sql_upper else ""
            }
            
            # Format SELECT part
            select_items = sql_parts["SELECT"].replace("SELECT", "").strip().split(",")
            sql_lines.append('<span style="color: #569cd6;">SELECT</span>')
            for item in select_items:
                # Check if this contains AS for aliasing
                if " AS " in item.upper():
                    parts = item.split(" AS ", 1)
                    field = parts[0].strip()
                    alias = parts[1].strip()
                    
                    if "." in field:
                        table_alias, column = field.split(".", 1)
                        sql_lines.append(f'    {table_alias}.<span style="color: white;">{column}</span> <span style="color: #569cd6;">AS</span> {alias},')
                    else:
                        sql_lines.append(f'    <span style="color: white;">{field}</span> <span style="color: #569cd6;">AS</span> {alias},')
                else:
                    field = item.strip()
                    if "." in field:
                        table_alias, column = field.split(".", 1)
                        sql_lines.append(f'    {table_alias}.<span style="color: white;">{column}</span>,')
                    else:
                        sql_lines.append(f'    <span style="color: white;">{field}</span>,')
            
            # Remove the trailing comma from the last SELECT item
            if sql_lines[-1].endswith(","):
                sql_lines[-1] = sql_lines[-1][:-1]
            
            # Format FROM part
            from_part = sql_parts["FROM"].replace("FROM", "").strip()
            sql_lines.append('<span style="color: #569cd6;">FROM</span>')
            if " JOIN " in from_part.upper():
                # Handle JOINs
                from_parts = from_part.split(" JOIN ")
                base_table = from_parts[0].strip()
                sql_lines.append(f'    {base_table}')
                
                for join_part in from_parts[1:]:
                    join_condition = join_part.split(" ON ", 1)
                    if len(join_condition) == 2:
                        join_table = join_condition[0].strip()
                        on_clause = join_condition[1].strip()
                        
                        # Format the ON clause with the "=" properly highlighted
                        on_parts = on_clause.split("=")
                        if len(on_parts) == 2:
                            left_side = on_parts[0].strip()
                            right_side = on_parts[1].strip()
                            on_formatted = f'{left_side} <span style="color: #d7ba7d;">=</span> {right_side}'
                        else:
                            on_formatted = on_clause
                        
                        sql_lines.append(f'<span style="color: #569cd6;">JOIN</span>')
                        sql_lines.append(f'    {join_table} <span style="color: #569cd6;">ON</span> {on_formatted}')
                    else:
                        sql_lines.append(f'<span style="color: #569cd6;">JOIN</span> {join_part}')
            else:
                sql_lines.append(f'    {from_part}')
            
            # Format WHERE part if it exists
            if sql_parts["WHERE"]:
                where_part = sql_parts["WHERE"].replace("WHERE", "").strip()
                sql_lines.append('<span style="color: #569cd6;">WHERE</span>')
                
                # Handle common WHERE conditions
                conditions = []
                if " AND " in where_part:
                    conditions = where_part.split(" AND ")
                    for i, condition in enumerate(conditions):
                        if i > 0:
                            sql_lines.append('<span style="color: #d7ba7d;">AND</span>')
                        
                        # Format condition with proper highlighting
                        if "=" in condition:
                            parts = condition.split("=", 1)
                            left = parts[0].strip()
                            right = parts[1].strip()
                            
                            # Check if right side is a string literal
                            if right.startswith("'") and right.endswith("'"):
                                right = f'<span style="color: #ce9178;">{right}</span>'
                            
                            sql_lines.append(f'    {left} <span style="color: #d7ba7d;">=</span> {right}')
                        elif "BETWEEN" in condition.upper():
                            parts = condition.upper().split("BETWEEN", 1)
                            field = parts[0].strip()
                            range_part = parts[1].strip()
                            
                            if " AND " in range_part:
                                range_parts = range_part.split(" AND ", 1)
                                start_val = range_parts[0].strip()
                                end_val = range_parts[1].strip()
                                
                                # Check if values are string literals
                                if "'" in start_val:
                                    start_val = f'<span style="color: #ce9178;">{start_val.lower()}</span>'
                                if "'" in end_val:
                                    end_val = f'<span style="color: #ce9178;">{end_val.lower()}</span>'
                                
                                sql_lines.append(f'    {field.lower()} <span style="color: #d7ba7d;">BETWEEN</span> {start_val} <span style="color: #d7ba7d;">AND</span> {end_val}')
                            else:
                                sql_lines.append(f'    {field.lower()} <span style="color: #d7ba7d;">BETWEEN</span> {range_part.lower()}')
                        else:
                            sql_lines.append(f'    {condition}')
                else:
                    sql_lines.append(f'    {where_part}')
                
                # Add semicolon at the end if present
                if sql_query.strip().endswith(";"):
                    sql_lines[-1] += '<span style="color: #e6e6e6;">;</span>'
        except Exception as e:
            # If formatting fails, fall back to simpler highlighting
            sql_lines = [sql_query]
    else:
        sql_lines = [sql_query]
    
    # Combine SQL lines into final HTML
    sql_html = "<br>".join(sql_lines)
    
    # Create the HTML component
    css = """
    <style>
    .sql-display {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 20px;
        border-radius: 5px;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        line-height: 1.6;
        margin: 15px 0;
        font-size: 14px;
    }
    </style>
    """
    
    full_html = f"{css}<div class='sql-display'>{sql_html}</div>"
    
    # Display using components.html
    import streamlit.components.v1 as components
    components.html(full_html, height=70 + (len(sql_lines) * 25))

def format_result_text(result_text):
    """Format the result text for better display"""
    # Check if it's a simple count/number result about Machine Learning
    ml_pattern = re.search(r'(\d+)\s+users?.*(?:interested|like).*(?:Machine Learning|ML)', result_text, re.IGNORECASE)
    if ml_pattern:
        number = ml_pattern.group(1)
        return f"The number of users interested in Machine Learning is {number}."
    
    # For 'liked between dates' queries, try to extract a clean simple message
    likes_pattern = re.search(r'(\d+)\s+(?:items?|articles?|interactions?).*liked', result_text, re.IGNORECASE)
    if likes_pattern:
        return f"Found {likes_pattern.group(1)} liked items between the specified dates."
    
    # For other types of results, just return the first sentence if it's short enough
    first_sentence = result_text.split('.')[0]
    if len(first_sentence) < 100:
        return first_sentence + "."
    
    # Default case, return original text
    return result_text

def display_table(df):
    """Display a dataframe as a styled HTML table matching the screenshots"""
    # Define CSS for the table
    css = """
    <style>
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        background-color: #1e1e1e;
        color: white;
    }
    .result-table th {
        padding: 12px 15px;
        text-align: left;
        background-color: #1e1e1e;
        color: white;
        font-weight: bold;
        border-bottom: 1px solid #444;
    }
    .result-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #333;
    }
    .result-table tr:hover {
        background-color: #2a2a2a;
    }
    .result-table tr:last-child td {
        border-bottom: none;
    }
    </style>
    """
    
    # Generate HTML table
    table_html = '<table class="result-table"><thead><tr>'
    
    # Add headers
    for col in df.columns:
        table_html += f'<th>{col}</th>'
    table_html += '</tr></thead><tbody>'
    
    # Add rows
    for _, row in df.iterrows():
        table_html += '<tr>'
        for val in row:
            table_html += f'<td>{val}</td>'
        table_html += '</tr>'
    
    table_html += '</tbody></table>'
    
    # Combine CSS and HTML
    full_html = f"{css}{table_html}"
    
    # Display using components.html
    import streamlit.components.v1 as components
    components.html(full_html, height=100 + (len(df) * 50))

def display_python_code(result_text):
    """Display Python code blocks from the result text"""
    # Extract Python code blocks from the result text
    python_blocks = re.findall(r'```python(.*?)```', result_text, re.DOTALL)
    
    if not python_blocks:
        return False
    
    # CSS for the code display
    css = """
    <style>
    .python-code {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        line-height: 1.6;
        margin: 10px 0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .py-keyword { color: #569cd6; }
    .py-string { color: #ce9178; }
    .py-comment { color: #6a9955; }
    .py-function { color: #dcdcaa; }
    .py-class { color: #4ec9b0; }
    .py-number { color: #b5cea8; }
    </style>
    """
    
    # Format each Python code block with syntax highlighting
    for code_block in python_blocks:
        code = code_block.strip()
        
        # Simple syntax highlighting
        # Keywords
        keywords = ['import', 'from', 'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'try', 'except', 'with']
        for keyword in keywords:
            code = re.sub(rf'\b{keyword}\b', f'<span class="py-keyword">{keyword}</span>', code)
        
        # Strings
        code = re.sub(r'(\'.*?\'|\".*?\")', r'<span class="py-string">\1</span>', code)
        
        # Comments
        code = re.sub(r'(#.*)', r'<span class="py-comment">\1</span>', code)
        
        # Function names
        code = re.sub(r'def\s+(\w+)', r'<span class="py-keyword">def</span> <span class="py-function">\1</span>', code)
        code = re.sub(r'(\w+)\(', r'<span class="py-function">\1</span>(', code)
        
        # Class names
        code = re.sub(r'class\s+(\w+)', r'<span class="py-keyword">class</span> <span class="py-class">\1</span>', code)
        
        # Numbers
        code = re.sub(r'\b(\d+)\b', r'<span class="py-number">\1</span>', code)
        
        # Display the highlighted code
        st.markdown(f"{css}<div class='python-code'>{code}</div>", unsafe_allow_html=True)
    
    return True

def display_tokens(input_tokens, output_tokens, total_tokens):
    """Display token usage in a format matching the screenshots"""
    css = """
    <style>
    .token-container {
        width: 100%;
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
        background-color: #1e1e1e;
        padding: 20px 0;
        border-top: 1px solid #444;
        border-bottom: 1px solid #444;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    }
    .token-box {
        flex: 1;
        text-align: center;
    }
    .token-label {
        color: #888;
        font-size: 18px;
        margin-bottom: 5px;
    }
    .token-value {
        color: white;
        font-size: 36px;
        font-weight: normal;
    }
    .provider-info {
        color: white;
        margin-top: 20px;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    }
    .processing-time {
        color: #888;
        margin-top: 10px;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    }
    </style>
    """
    
    html = f"""
    <div class="token-container">
        <div class="token-box">
            <div class="token-label">Input Tokens</div>
            <div class="token-value">{input_tokens}</div>
        </div>
        <div class="token-box">
            <div class="token-label">Output Tokens</div>
            <div class="token-value">{output_tokens}</div>
        </div>
        <div class="token-box">
            <div class="token-label">Total Tokens</div>
            <div class="token-value">{total_tokens}</div>
        </div>
    </div>
    """
    
    # Combine CSS and HTML
    full_html = f"{css}{html}"
    
    # Display using components.html
    import streamlit.components.v1 as components
    components.html(full_html, height=120)

# Main layout
# Replace the title with a simpler version
st.markdown("<h2 style='margin-bottom:20px'>🤖 SQL Agent Interface</h2>", unsafe_allow_html=True)

# Make the sidebar more compact and less distracting
with st.sidebar:
    st.markdown("<h3>Settings</h3>", unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider",
        ["openai", "ollama"],
        index=0 if "provider" not in st.session_state else 0 if st.session_state.provider == "openai" else 1
    )

    if provider == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        model = st.selectbox("Model", model_options, index=0)
    else:  # ollama
        model_options = [
            "qwen2.5-coder:32b",
            "qwen2.5-coder",
            "qwen2.5-coder:3b",
            "llama3.3",
            "mistral-small3.1"
        ]
        model = st.selectbox("Model", model_options, index=0)
    
    st.divider()
    st.markdown("<small>This interface runs SQL queries using AI</small>", unsafe_allow_html=True)

# Main layout
user_query = st.text_area("Enter your query:", 
                          value="what has been liked between '2023-04-13' and '2023-04-15 11:00:00'\ngive the results in a bar chart, user name in axis x, hour of likes in y axis and color by item name (python code)" 
                          if "query" not in st.session_state else st.session_state.query,
                          height=80)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Run Query", type="primary"):
        with st.spinner(f"Running query with {provider.capitalize()}..."):
            st.session_state.provider = provider
            st.session_state.query = user_query
            success, result = run_query(provider, user_query)
            
            if success:
                st.session_state.run_complete = True
                st.session_state.result_file = result if isinstance(result, str) else None
                st.rerun()
            else:
                st.error(result)

# Display results if available
if "run_complete" in st.session_state and st.session_state.run_complete:
    result_file = st.session_state.get("result_file")
    result_data = read_latest_result(result_file)
    
    if result_data:
        # Get the result text and SQL
        result_text = result_data.get("Result", "")
        sql_query = result_data.get("SQL", "")
        
        # Display the "Result" section header
        st.markdown("## Result")
        
        # Simply display the full result text with markdown formatting
        # This preserves the exact format from the markdown file
        st.markdown(result_text)
        
        # Display the "SQL" section header
        st.markdown("## SQL")
        
        # Display the SQL query
        display_sql(sql_query)
        
        # Check if there's a table in the result text and display it
        df = parse_table_data(result_text)
        if df is not None:
            st.markdown("### Data Table")
            display_table(df)
        
        # Display token information and footer only if requested
        if st.sidebar.checkbox("Show Token Usage and Processing Info", value=True):
            # Token usage
            token_usage = result_data.get("Token Usage", "")
            token_lines = token_usage.strip().split("\n")
            
            input_tokens = "N/A"
            output_tokens = "N/A"
            total_tokens = "N/A"
            
            for line in token_lines:
                if "Input tokens:" in line:
                    input_tokens = line.split(":")[1].strip()
                elif "Output tokens:" in line:
                    output_tokens = line.split(":")[1].strip()
                elif "Total tokens:" in line:
                    total_tokens = line.split(":")[1].strip()
            
            # Display token information
            display_tokens(input_tokens, output_tokens, total_tokens)
            
            # Provider and model info
            provider_info = result_data.get("Provider", "")
            provider_lines = provider_info.strip().split("\n")
            
            provider_name = "N/A"
            model_name = "N/A"
            
            for line in provider_lines:
                if "Provider:" in line:
                    provider_name = line.split(":")[1].strip()
                elif "Model:" in line:
                    model_name = line.split(":")[1].strip()
            
            st.markdown(f"""
            <div style="font-family: 'Menlo', 'Monaco', 'Courier New', monospace; color: white; margin-bottom: 5px;">
            Query processed with <b>{provider_name}</b> using model <b>{model_name}</b>
            </div>
            """, unsafe_allow_html=True)
            
            # Processing time
            time_info = result_data.get("Processing Time", "")
            time_lines = time_info.strip().split("\n")
            
            begin_time = "N/A"
            end_time = "N/A"
            
            for line in time_lines:
                if "Begin:" in line:
                    begin_time = line.split("Begin:")[1].strip()
                elif "End:" in line:
                    end_time = line.split("End:")[1].strip()
            
            # Directly display the full timestamp strings without any parsing
            # Get result filename from path
            result_filename = "N/A"
            if result_file:
                result_filename = os.path.basename(result_file)
            
            st.markdown(f"""
            <div style="font-family: 'Menlo', 'Monaco', 'Courier New', monospace; color: #888;">
            Processing time: {begin_time} to {end_time}<br>
            Result file: {result_filename}
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.error("No results found. Please run a query first.")
else:
    st.info("Enter a query and click 'Run Query' to get started.")