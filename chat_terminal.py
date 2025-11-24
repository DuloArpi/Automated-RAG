import requests
import sys

API_URL = "http://localhost:8000/query"

print("--- AutoRAG Terminal Chat ---")
print("Type 'exit' to quit.\n")

while True:
    try:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        # Send to server
        response = requests.post(API_URL, json={"question": question})
        
        if response.status_code == 200:
            data = response.json()
            print(f"Bot: {data['answer']}")
            print(f"\n[Sources used: {len(data['sources'])} chunks]")
        else:
            print(f"Error: {response.text}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Connection Error: {e}")
        break