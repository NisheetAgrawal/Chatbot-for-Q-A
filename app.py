from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Your existing code here, including routes and any other configurations
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle cross-origin requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load FAQ data
data = pd.read_csv('upYOG_faqs.csv')

# Initialize the model and generate embeddings for each FAQ question
model = SentenceTransformer('all-MiniLM-L6-v2')
prompt_embeddings = model.encode(data['prompt'].tolist())

# Initialize FAISS index and add embeddings
dimension = prompt_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(prompt_embeddings.astype(np.float32))

# Define the function to retrieve the answer based on the query
def retrieve_document(query):
    query_embedding = model.encode([query])[0]
    _, idx = index.search(np.array([query_embedding], dtype=np.float32), k=1)
    return data.iloc[idx[0][0]]['response']

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the API route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    response = retrieve_document(query)
    return jsonify({'query': query, 'response': response})

# Run the Flask app
if __name__ == "__main__":
    # Set to a unique port to avoid conflicts; 7001 in this example
    app.run(port=7001, host="0.0.0.0")
