#Build a Retrieval-Augmented Generation (RAG) system using Hugging Face's Transformers library.
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents (replace this with your dataset)
documents = [
    "Python is a high-level programming language.",
    "Transformers library by Hugging Face provides easy access to state-of-the-art NLP models.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Retrieval-Augmented Generation combines retrieval and generation for better responses."
]

# Step 1: Index Documents Using FAISS
def create_index(documents):
    # Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Convert TF-IDF matrix to dense numpy array
    dense_matrix = tfidf_matrix.toarray().astype(np.float32)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dense_matrix.shape[1])
    index.add(dense_matrix)
    
    return index, vectorizer

# Create index and vectorizer
index, vectorizer = create_index(documents)

# Step 2: Load RAG Model and Tokenizer
tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', index=index, vectorizer=vectorizer)
model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq')

# Step 3: Define Retrieval and Generation Functions
def retrieve_and_generate(query):
    # Retrieve relevant documents
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = index.search(query_vector, k=2)  # Retrieve top 2 documents
    
    # Combine retrieved documents
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)
    
    # Tokenize and generate response
    inputs = tokenizer([query], return_tensors='pt')
    context_inputs = tokenizer([context], return_tensors='pt')
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        context_input_ids=context_inputs['input_ids'],
        context_attention_mask=context_inputs['attention_mask']
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 4: Test the System
query = "What is FAISS used for?"
response = retrieve_and_generate(query)
print(f"Query: {query}")
print(f"Response: {response}")
