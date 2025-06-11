from dotenv import load_dotenv
import os
import argparse
from openai import OpenAI
import importlib
from src.sql_agents_common import run_demo_loop

load_dotenv()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SQL agent with OpenAI or Ollama')
    parser.add_argument('--provider', type=str, choices=['openai', 'ollama'], default='openai',
                        help='Provider to use: openai or ollama (default: openai)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Dynamically import the appropriate modules based on provider
    provider_module = importlib.import_module(f"src.sql_agents_{args.provider}")
    
    # Get router agent and token counter from the imported module
    sql_router_agent = provider_module.sql_router_agent
    token_counter = provider_module.token_counter
    
    if args.provider == 'openai':
        # Initialize OpenAI client
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        provider_name = "OpenAI"
    else:  # ollama
        # Initialize Ollama client
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        provider_name = "Ollama"
    
    # Run the demo loop
    run_demo_loop(
        client=client,
        starting_agent=sql_router_agent,
        token_counter=token_counter,
        provider_name=provider_name
    )

if __name__ == "__main__":
    main() 