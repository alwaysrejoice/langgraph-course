from dotenv import load_dotenv
import os

load_dotenv()
from graph.graph import app


def main():
    print("Hello from agentic-rag!")
    print(app.invoke(input={"question": "what is agent memory?"}))

if __name__ == "__main__":
    main()
