# Healthcare Bot for Pathological Lab Report Evaluation

## Introduction
This Python script is a healthcare bot designed for creating a robust generative search system for pathological lab reports in PDF format. The system aims to effectively answer user queries related to lab reports using various components such as chunking strategies, embedding models, re-rankers, and generation prompts.

## Setup Instructions
To use the tool, follow the instructions below:

### Prerequisites
1. Python 3.x
2. pip (Python package installer)

### Installation Steps
1. Clone the repository:
    ```bash
    git clone git@github.com:jpjayprasad-dev/healthcare-bot.git
    cd healthcare-bot
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Obtain OpenAI API Key:
    Set your OpenAI API key as an environment variable named OPENAI_KEY or store it in a file named open_ai_key.txt.

4. Run the tool:
    ```bash
    python main.py
    ```
### Usage
Lab Report Processing:
The script processes PDF lab reports, extracts text, and organizes it into a DataFrame.

User Queries:
Enter the path of the lab report PDF when prompted.
After processing, the tool awaits user queries.

Query Processing:
Enter your query to retrieve relevant information from the lab report.

Output:
The tool provides informative answers to user queries, leveraging components like semantic search, re-ranking, and GPT-3.5 for generating responses.

Exiting:
Type 'exit' to exit the tool.

### Components
PDF Processing:
Utilizes the pdfplumber library to extract text and tables from PDFs.

Semantic Search:
Utilizes ChromaDB for storing and searching document vectors based on text embeddings.

Re-ranking:
Uses a cross-encoder model to re-rank search results.

Response Generation:
Uses GPT-3.5-turbo for generating informative responses.

### Dependencies
    pdfplumber
    pandas
    tiktoken
    openai
    chromadb
    sentence_transformers
