from pymilvus import MilvusClient
import json
from tqdm import tqdm
from glob import glob
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Reading markdown files
text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()
    text_lines += file_text.split("# ")

def emb_text(text):
    """Generate embedding for the given text using OpenAI API"""
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

# Test embedding to determine the dimension
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
# print(embedding_dim)
# print(test_embedding[:10])

# Connect to Milvus
milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

collection_name = "my_rag_collection"

# Drop collection if exists and create a new one
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

# Prepare data for insertion
data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)

# Define the question
question = "How is data stored in milvus?"

# Perform search in Milvus
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)

# Process search results
retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
# print(json.dumps(retrieved_lines_with_distances, indent=4))

# Combine context from retrieved lines
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

# Prepare prompts for OpenAI chat model
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

# Get response from OpenAI model
response = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
