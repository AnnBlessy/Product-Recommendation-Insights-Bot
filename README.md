# Product-Recommendation-Insights-Bot
The Product Recommendation Insights Bot is a data-driven chatbot tailored for e-commerce businesses specializing in technical products such as Apple, HP, Dell, and similar brands. It enhances business strategies by offering personalized product recommendations and market insights based on customer purchasing data. By analyzing customer preferences and shopping patterns for technical products, the bot provides relevant suggestions rooted in past purchases and seasonal trends. This enables businesses to optimize inventory, refine marketing strategies, and enhance overall customer engagement. The bot effectively responds to queries about popular technical products and related recommendations, helping businesses increase sales and improve customer satisfaction.

The step-by-step guide for integrating PDF processing into your Retrieval-Augmented Generation (RAG) based chatbot are:

### Step 1: Install Required Libraries

```
pip install sentence-transformers faiss-cpu PyMuPDF
```

### Step 2: Extract Text from PDFs

```
# Step 1: Extract Text from PDF File
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()  # Extract text from each page
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()
def split_text(text):
    return sent_tokenize(text)  # Split text into sentences using NLTK
```

### Step 3: Create Embeddings
Create Embeddings for each sentence/paragraph

```
def create_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks, convert_to_numpy=True)  # Create embeddings for each text chunk
    return embeddings
```

### Step 4: Store Embeddings with FAISS
Next, store these embeddings in a FAISS index for fast retrieval.

```
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
    index = faiss.IndexFlatIP(dimension)  # Using Inner Product (Cosine Similarity)
    faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
    index.add(embeddings)  # Add the embeddings to the index
    return index
```

### Step 5: Query and Retrieve Relevant Text
Now, set up a function to retrieve the most relevant texts based on user queries.

```
def retrieve_similar_sentences(query, model, index, text_chunks, threshold=0.5):
    query_embedding = model.encode([query], convert_to_numpy=True)  # Create an embedding for the query
    faiss.normalize_L2(query_embedding)  # Normalize the query embedding for cosine similarity
    distances, indices = index.search(query_embedding, 3)  # Search for the top 3 nearest embeddings
    results = [text_chunks[i] for i, distance in zip(indices[0], distances[0]) if distance > threshold]  # Apply threshold
    return results if results else ["No relevant sentences found."]
```

### Step-6: Curate the Responses
Curate Response Based on System Message

```
def curate_response(similar_sentences, system_message):
    if system_message == "summarize":
        summarizer = pipeline("summarization")  # Use a summarization model
        summarized_text = summarizer(' '.join(similar_sentences), max_length=130, min_length=30, do_sample=False)
        return summarized_text[0]['summary_text']

    elif system_message == "detailed response":
        return '\n\n'.join(similar_sentences)

    elif system_message == "insights only":
        insights = [sentence for sentence in similar_sentences if sentence.startswith('-')]  # Extract bullet points
        return '\n'.join(insights) if insights else "No key insights found."

    else:
        return '\n\n'.join(similar_sentences)
```

### Step 7: Putting It All Together (Main Function)

```
def main():
    # Step 1: Extract text from a PDF file
    pdf_file = '/content/apple-products.pdf'
    raw_text = extract_text_from_pdf(pdf_file)

    # Step 2: Clean and preprocess the text
    cleaned_text = clean_text(raw_text)

    # Step 3: Split the text into sentences
    text_chunks = split_text(cleaned_text)

    # Step 4: Load the embedding model and create embeddings for each sentence
    model = SentenceTransformer('paraphrase-mpnet-base-v2')  # A larger, more robust model for better embeddings
    embeddings = create_embeddings(text_chunks, model)

    # Step 5: Build the FAISS index
    index = build_faiss_index(embeddings)

    print("Welcome to the RAG Bot!")
    print("Ask your question, or type 'exit' to quit.")
    system_message = 'detailed response'

    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        # Step 6: Retrieve relevant sentences based on the query
        similar_sentences = retrieve_similar_sentences(query, model, index, text_chunks)

        # Step 7: Curate the response based on system message
        curated_response = curate_response(similar_sentences, system_message)

        # Display result
        print(f"Bot:\n{curated_response}\n")

if __name__ == "__main__":
    main()
```

### Summary:
- **Extract Text**: The `extract_text_from_pdfs` function reads all PDF files in the specified folder and extracts their text.
- **Create Embeddings**: The `create_embeddings` function generates embeddings for each document's text.
- **Store in FAISS**: The `build_faiss_index` function stores the embeddings for quick retrieval.
- **Querying**: The bot retrieves and displays relevant documents based on user queries.

## OUTPUT:

![image](https://github.com/user-attachments/assets/223c05fd-c8c8-4e33-aa3a-0fcdd0381af5)

![image](https://github.com/user-attachments/assets/b82ab3c4-9bc0-469c-adfd-5a03e082ac07)


