import boto3
import json
from pymilvus import (
    connections,
    Collection,
    utility
)
import numpy as np

# AWS Bedrock Configuration
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1'
)

def connect_to_milvus(host="127.0.0.1", port="19530"):
    """Establish connection to Milvus"""
    try:
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print("Successfully connected to Milvus")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def get_text_embedding_from_bedrock(text):
    """Get text embeddings from Amazon Bedrock"""
    output_embedding_length = 512
    response = bedrock_client.invoke_model(
        modelId="cohere.embed-english-v3",
        contentType="application/json",
        accept="*/*",
        body=json.dumps({
            "texts": [text],
            "input_type": "search_document"
        })
    )
    response_body = json.loads(response.get('body').read())
    embeddings = []
    for embedding in response_body.get('embeddings'):
        embeddings.append(embedding)
    return embeddings[0]

def search_collection(collection_name, query_text, limit=3):
    """
    Search the collection using both vector similarity and text matching
    """
    try:
        # Get collection
        collection = Collection(collection_name)
        collection.load()

        # Get embedding for query text
        query_embedding = get_text_embedding_from_bedrock(query_text)

        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # Perform hybrid search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text", "file_name", "chunk_id"]  # Add any other fields you want to retrieve
        )

        # Process and print results
        if results:
            for i, hits in enumerate(results):
                print(f"\nSearch Results:")
                for hit in hits:
                    print("\n---")
                    print(f"Distance: {hit.distance}")
                    if hit.entity.get('text'):
                        print(f"Text: {hit.entity.get('text')}")
                    if hit.entity.get('file_name'):
                        print(f"File: {hit.entity.get('file_name')}")
                    if hit.entity.get('chunk_id') is not None:
                        print(f"Chunk ID: {hit.entity.get('chunk_id')}")
        else:
            print("No results found")

        # Release collection
        collection.release()

    except Exception as e:
        print(f"Search failed: {e}")
        raise

def main():
    # Connect to Milvus
    connect_to_milvus()

    # Your query text
    query_text = """i used to go with natalie , but when she and ben got serious , i declined her invitations to spend the holidays with her family . it felt weird taggingalong and i wanted to give them some space . natalie always begged me to join them , but i lied and told her i would be fine and already had plans here . with apangin my chest , i remembered the first time i spent christmas alone . i sat on my couch the whole day , watched a channel that played a christmas story nonstop , and bawled like a baby . after that , i went to the soup kitchen on holidays . i more or less came to terms with not having any family , but the fact that no one except natalie would notice if i died brought on the bout of depression . it was pathetic that i did n't have any other friends that i could spend the holidays with . absolutely pathetic . the serving spoon shook in my hands . i ca n't spend the rest of my life like this . the sting of tears threatened . in a few years i 'll be thirty . the soup kitchen faded away as depression wrapped its coils around my chest like a python , squeezing me of air . i gave up years ago on a happy , picture-perfect life , but it was hard to bear this"""

    # Search collection
    search_collection("TitalMulti_1", query_text, limit=3)

    # Close connection
    connections.disconnect("default")

if __name__ == "__main__":
    main()
