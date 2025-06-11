import streamlit as st
import os
import subprocess
import glob
import time
import re
from datetime import datetime
import shlex

def extract_sections(content, sections=["## SQL", "## Result"]):
    """Extract specific sections from the markdown content"""
    results = {}
    
    # Create a pattern to match section headers and capture content until the next section
    pattern = r'(## [^\n]+)(.*?)(?=## |$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for header, section_content in matches:
        header = header.strip()
        if header in sections:
            results[header] = section_content.strip()
    
    return results

def run_query(provider, query):
    """Run the query using run.py and return the path to the output file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the count of query files before running
    initial_files = set(glob.glob("output/query_*.md"))
    initial_result_files = set(glob.glob("output/results/query_*.md"))
    
    # Check both output directories
    all_initial_files = initial_files.union(initial_result_files)
    
    try:
        # Use a simpler approach: run the command directly with input
        cmd = ["python", "run.py", f"--provider={provider}"]
        
        # Run the command and pipe input
        process = subprocess.run(
            cmd,
            input=query + "\n",  # Add newline to simulate pressing Enter
            text=True,
            capture_output=True
        )
        
        stdout = process.stdout
        stderr = process.stderr
        
        # Wait a bit longer for file writing to complete
        time.sleep(3)
        
        # Check for new output files in both locations
        current_files = set(glob.glob("output/query_*.md"))
        current_result_files = set(glob.glob("output/results/query_*.md"))
        all_current_files = current_files.union(current_result_files)
        
        new_files = all_current_files - all_initial_files
            
        if new_files:
            # Return the path to the most recent query file
            newest_file = max(new_files, key=os.path.getctime)
            return newest_file, stdout
        else:
            # If no new files were found, return the console output
            return None, f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            
    except Exception as e:
        return None, f"Error running query: {str(e)}"

def display_query_sections(content):
    """Display only SQL and Result sections from query output"""
    sections = extract_sections(content)
    
    if "## SQL" in sections:
        st.markdown("### SQL Query")
        st.code(sections["## SQL"], language="sql")
    
    if "## Result" in sections:
        st.markdown("### Results")
        st.markdown(sections["## Result"])
    
    if not sections:
        st.warning("No SQL or Result sections found in the output file.")
        st.markdown("### Full Output")
        st.markdown(content)

def main():
    st.set_page_config(
        page_title="SQL Agents with Multiple LLM Providers",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("SQL Agents with Multiple LLM Providers")
    st.markdown("A flexible system for querying databases using natural language.")
    
    # Create the output directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/results", exist_ok=True)
    
    # Sidebar provider selection
    st.sidebar.header("Provider Settings")
    provider = st.sidebar.radio(
        "Select LLM Provider",
        options=["openai", "ollama"],
        index=0,
        format_func=lambda x: "OpenAI" if x == "openai" else "Ollama"
    )
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "sections" in message:
                # Display structured sections
                if "## SQL" in message["sections"]:
                    st.markdown("### SQL Query")
                    st.code(message["sections"]["## SQL"], language="sql")
                
                if "## Result" in message["sections"]:
                    st.markdown("### Results")
                    st.markdown(message["sections"]["## Result"])
            else:
                # Display regular content
                st.markdown(message["content"])
    
    # Show previous query files
    st.sidebar.markdown("---")
    st.sidebar.header("Previous Queries")
    # Check both possible output locations
    query_files = sorted(glob.glob("output/query_*.md"), reverse=True)
    result_files = sorted(glob.glob("output/results/query_*.md"), reverse=True)
    all_query_files = sorted(query_files + result_files, key=os.path.getctime, reverse=True)
    
    selected_file = st.sidebar.selectbox(
        "View Previous Queries", 
        options=all_query_files,
        format_func=lambda x: x.split("/")[-1] if x else "None",
        index=0 if all_query_files else None
    )
    
    if selected_file and st.sidebar.button("Load Selected Query"):
        try:
            with open(selected_file, 'r') as f:
                content = f.read()
            
            sections = extract_sections(content)
            
            st.sidebar.markdown("### Query Details")
            if sections:
                if "## SQL" in sections:
                    st.sidebar.markdown("**SQL Query:**")
                    st.sidebar.code(sections["## SQL"], language="sql")
                
                if "## Result" in sections:
                    st.sidebar.markdown("**Results:**")
                    st.sidebar.markdown(sections["## Result"])
            else:
                st.sidebar.text_area("Full Content", value=content, height=300, disabled=True)
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Chat input
    if query := st.chat_input("Ask a question about your database..."):
        # Display user query
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process the query with the selected provider
        with st.chat_message("assistant"):
            with st.spinner(f"Running query using {provider.capitalize()}..."):
                output_file, console_output = run_query(provider, query)
                
                # Show results
                if output_file:
                    try:
                        with open(output_file, 'r') as f:
                            content = f.read()
                        
                        # Extract and display only SQL and Result sections
                        sections = extract_sections(content)
                        
                        if sections:
                            if "## SQL" in sections:
                                st.markdown("### SQL Query")
                                st.code(sections["## SQL"], language="sql")
                            
                            if "## Result" in sections:
                                st.markdown("### Results")
                                st.markdown(sections["## Result"])
                            
                            # Save both the full content and sections
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": content,
                                "sections": sections
                            })
                        else:
                            st.warning("No SQL or Result sections found in the output file.")
                            st.markdown(content)
                            st.session_state.messages.append({"role": "assistant", "content": content})
                    except Exception as e:
                        error_message = f"Error reading output file: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                else:
                    st.error("No output file was generated. See console output below:")
                    st.code(console_output, language="bash")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {console_output}"})
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This application uses language models to interact with your database using natural language."
    )

if __name__ == "__main__":
    main() 