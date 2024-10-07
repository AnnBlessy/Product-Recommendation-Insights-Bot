# Customer-Feedback-Insights-Bot
The Product Recommendation Insights Bot is a data-driven chatbot tailored for e-commerce businesses specializing in technical products such as Apple, HP, Dell, and similar brands. It enhances business strategies by offering personalized product recommendations and market insights based on customer purchasing data. By analyzing customer preferences and shopping patterns for technical products, the bot provides relevant suggestions rooted in past purchases and seasonal trends. This enables businesses to optimize inventory, refine marketing strategies, and enhance overall customer engagement. The bot effectively responds to queries about popular technical products and related recommendations, helping businesses increase sales and improve customer satisfaction.

The step-by-step guide for integrating PDF processing into your Retrieval-Augmented Generation (RAG) based chatbot are:

1. **Extracting Text from PDFs** using the `PyPDF2` library.
2. **Creating Embeddings** using a `SentenceTransformer`.
3. **Storing Embeddings** with FAISS for quick retrieval.
4. **Retrieving Relevant Text** based on user queries.

### Step 1: Install Required Libraries

```
pip install PyPDF2 sentence-transformers faiss-cpu
```

### Step 2: Extract Text from PDFs

```
import PyPDF2
import os

def extract_text_from_pdfs(pdf_folder):
    pdf_texts = {}
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, filename)
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                pdf_texts[filename] = text.strip()  # Store the text with the filename as the key
                
    return pdf_texts
```

### Step 3: Create Embeddings
Using `SentenceTransformer`, we can create embeddings from the extracted text.

```
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(texts):
    # Create embeddings for each text entry
    embeddings = model.encode(list(texts.values()), convert_to_numpy=True)
    return embeddings
```

### Step 4: Store Embeddings with FAISS
Next, store these embeddings in a FAISS index for fast retrieval.

```
import faiss

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Create a flat L2 index
    index.add(embeddings)  # Add embeddings to the index
    return index
```

### Step 5: Query and Retrieve Relevant Text
Now, set up a function to retrieve the most relevant texts based on user queries.

```
def retrieve_similar_documents(query, model, index, texts, top_k=5):
    # Create an embedding for the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Perform the FAISS search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the documents based on indices
    similar_docs = {list(texts.keys())[i]: texts[list(texts.keys())[i]] for i in indices[0]}
    
    return similar_docs
```

### Step 6: Putting It All Together (Main Function)

```
def main():
    # Step 1: Extract text from PDFs
    pdf_folder = 'path/to/pdf_folder'  # Replace with your PDF folder path
    pdf_texts = extract_text_from_pdfs(pdf_folder)
    
    # Step 2: Create embeddings for the extracted texts
    embeddings = create_embeddings(pdf_texts)
    
    # Step 3: Build the FAISS index
    index = build_faiss_index(embeddings)
    
    print("Welcome to the Product Recommendation Insights Bot!")
    print("Ask your question, or type 'exit' to quit.")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break
        
        # Step 4: Retrieve relevant documents based on the query
        similar_documents = retrieve_similar_documents(query, model, index, pdf_texts)

        # Display results
        for doc_title, doc_content in similar_documents.items():
            print(f"\nDocument: {doc_title}\nContent: {doc_content[:500]}...")  # Display the first 500 characters of the content

if __name__ == "__main__":
    main()
```

### Summary:
- **Extract Text**: The `extract_text_from_pdfs` function reads all PDF files in the specified folder and extracts their text.
- **Create Embeddings**: The `create_embeddings` function generates embeddings for each document's text.
- **Store in FAISS**: The `build_faiss_index` function stores the embeddings for quick retrieval.
- **Querying**: The bot retrieves and displays relevant documents based on user queries.

**Fine-Tune the Model** - Depending on the responses, we need to further refine the model or the input data to enhance the relevance of the retrieved content.
