import sys
import os
from dotenv import load_dotenv

from travel_agent.agent import CCMAgent
from travel_agent.baseline_agent import BaselineAgent
from travel_agent.tools import reset_budget

# Load env variables for API keys
load_dotenv()

def main():
    print("==================================================")
    print("         Travel Agent Interactive Chat")
    print("==================================================")
    print("Choose the agent you want to chat with:")
    print("  1. CCM Agent (Context Compression Module)")
    print("  2. Baseline Agent (No compression, full history)")
    print("==================================================")
    
    try:
        choice = input("Enter 1 or 2: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)
    
    if choice == '1':
        print("\nInitializing CCM Agent (this may take a few seconds)...")
        agent = CCMAgent()
    elif choice == '2':
        print("\nInitializing Baseline Agent...")
        agent = BaselineAgent()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
        sys.exit(1)
        
    print("Resetting memory and budget...")
    agent.reset()
    
    print("\n" + "="*50)
    print("Agent is ready! Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            print("\nThinking (and potentially calling tools)...")
            
            # Chat with the agent
            result = agent.chat(user_input)
            
            print("\n" + "-"*50)
            print(f"Agent: {result.get('response', '')}")
            print("-"*50)
            print(f"Tokens in context this turn: {result.get('tokens_in_context', 0)}")
            print(f"Turn number: {result.get('turn_number', 0)}")
            
            # If CCM agent, optionally show memory stats
            if choice == '1' and 'memory_state' in result:
                stats = result['memory_state'].get('compression_stats', {})
                if stats and stats.get("total_tool_calls_compressed", 0) > 0:
                    avg_comp = stats.get("overall_compression_ratio", 0)
                    print(f"Average compression ratio: {avg_comp}x")
                    
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set in your environment or .env file.")
        print("The agent will likely fail to respond. Please set it and try again.")
    main()
