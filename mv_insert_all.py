import boto3
import json
import os
from PIL import Image
from io import BytesIO
import uuid
import base64
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import numpy as np

# AWS Bedrock Configuration
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1'
)

# Milvus Configuration
collection_name = "TitalMulti_1"

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

def create_milvus_collection(collection_name, dim=256):
    """Create Milvus collection with schema"""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'")

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="image", dtype=DataType.VARCHAR, max_length=65535),  # Max 10MB for base64 image
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Multi-modal collection for text and images"
    )

    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default'
    )

    # Create IVF_FLAT index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    print(f"Created collection '{collection_name}' with schema and index")
    return collection
#cohere.embed-english-v3
#amazon.titan-embed-image-v1
def get_image_embedding_from_bedrock(image):
    """Get image embeddings from Amazon Bedrock"""
    output_embedding_length = 512
    response = bedrock_client.invoke_model(
        modelId="cohere.embed-english-v3",
        contentType="application/json",
        accept="*/*",
        body=json.dumps({
            "inputImage": image,
            "embeddingConfig": {"outputEmbeddingLength": output_embedding_length}
        })
    )
    response_body = json.loads(response.get('body').read())
    return response_body["embedding"]

def get_text_embedding_from_bedrock(text):
    """Get text embeddings from Amazon Bedrock"""
    output_embedding_length = 256
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-image-v1",
        contentType="application/json",
        accept="*/*",
        body=json.dumps({
            "inputText": text,
            "embeddingConfig": {"outputEmbeddingLength": output_embedding_length}
        })
    )
    response_body = json.loads(response.get('body').read())
    return response_body["embedding"]

def read_file(file_path):
    """Read content from a file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=50):
    """Chunk text into smaller parts"""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def image_to_base64_data_url(image_path):
    """Convert image to base64 data URL"""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def main():
    # Connect to Milvus
    connect_to_milvus()

    # Create collection
    collection = create_milvus_collection(collection_name)

    # Directory paths
    text_path = '/home/ec2-user/milvus/tesdata/data/text/'
    img_path = '/home/ec2-user/milvus/tesdata/data/image/'

    # Get list of files
    files = [f for f in os.listdir(text_path) if os.path.isfile(os.path.join(text_path, f))]
    #img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

   # Process images
   # for file_name in img_files:
   #     file_path = os.path.join(img_path, file_name)

        # Convert image to base64
   #     img_data = image_to_base64_data_url(file_path)

        # Get embedding
   #     embedding = get_image_embedding_from_bedrock(img_data)

        # Insert into Milvus
     #   data = [
     #       [str(uuid.uuid4())],  # id
      #      [""],                 # file_name
      #      [0],                  # chunk_id
      #      [file_name],          # image_name
      #      [""],                 # text
       #     [img_data],           # image
       #     [embedding]           # embedding
       # ]

      #  collection.insert(data)
      #  print(f"Inserted image file '{file_name}'")

    # Process text files (maximum 10 files)
    for file_index, file_name in enumerate(files[:10]):
        file_path = os.path.join(text_path, file_name)

        # Read and chunk document
        document_text = read_file(file_path)
        chunks = list(chunk_text(document_text, chunk_size=250))

        for idx, chunk in enumerate(chunks):
            # Get embedding
            embedding = get_text_embedding_from_bedrock(chunk)

            # Insert into Milvus
            data = [
                [str(uuid.uuid4())],  # id
                [file_name],          # file_name
                [idx],                # chunk_id
                [""],                 # image_name
                [chunk],              # text
                [""],                 # image
                [embedding]           # embedding
            ]

            collection.insert(data)
            print(f"Inserted chunk {idx} from file '{file_name}'")

    # Flush to ensure all data is written
    collection.flush()

    # Print final statistics
    print(f"\nInsertion completed:")
    print(f"Total records in collection: {collection.num_entities}")

    # Close connection
    connections.disconnect("default")

if __name__ == "__main__":
    main()
