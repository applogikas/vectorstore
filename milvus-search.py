import boto3
import json
from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType
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

def create_collection(collection_name, dim=4096):
    """Create Milvus collection with correct dimension"""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'")

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Text collection with Cohere embeddings"
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
    
    print(f"Created collection '{collection_name}' with {dim}-dimensional vectors")
    return collection

def get_text_embedding_from_bedrock(text):
    """Get text embeddings from Amazon Bedrock"""
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
    embeddings = response_body.get('embeddings', [])
    if not embeddings:
        raise ValueError("No embeddings returned from Bedrock")
    return embeddings[0]

def search_collection(collection_name, query_text, limit=3):
    """
    Search the collection using vector similarity
    """
    try:
        # Get collection
        collection = Collection(collection_name)
        collection.load()

        # Get embedding for query text
        query_embedding = get_text_embedding_from_bedrock(query_text)
        
        # Verify embedding dimension
        embedding_dim = len(query_embedding)
        print(f"Query embedding dimension: {embedding_dim}")

        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # Perform search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text", "file_name", "chunk_id"]
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

def migrate_collection_data(old_collection_name, new_collection_name):
    """
    Migrate data from old collection to new collection with correct dimensions
    """
    try:
        # Get old collection
        old_collection = Collection(old_collection_name)
        old_collection.load()
        
        # Get new collection
        new_collection = Collection(new_collection_name)
        
        # Get all data from old collection
        print(f"\nReading data from old collection '{old_collection_name}'...")
        total_entities = old_collection.num_entities
        print(f"Total entities to migrate: {total_entities}")
        
        # Batch size for processing
        batch_size = 100
        
        for offset in range(0, total_entities, batch_size):
            # Adjust limit for last batch
            current_batch_size = min(batch_size, total_entities - offset)
            
            # Query batch of data from old collection
            data = old_collection.query(
                expr="id != ''",
                output_fields=["id", "file_name", "chunk_id", "text"],
                limit=current_batch_size,
                offset=offset
            )
            
            if not data:
                continue
                
            # Prepare data for new collection
            batch_texts = [item.get('text', '') for item in data if item.get('text')]
            
            if not batch_texts:
                continue
                
            # Generate new embeddings for the batch
            print(f"Generating embeddings for batch {offset//batch_size + 1}...")
            batch_embeddings = []
            for text in batch_texts:
                try:
                    embedding = get_text_embedding_from_bedrock(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error generating embedding: {e}")
                    continue
            
            # Prepare insert data
            insert_data = [
                [str(item.get('id', '')) for item in data],
                [str(item.get('file_name', '')) for item in data],
                [int(item.get('chunk_id', 0)) for item in data],
                batch_texts,
                batch_embeddings
            ]
            
            # Insert batch into new collection
            try:
                new_collection.insert(insert_data)
                print(f"Inserted batch {offset//batch_size + 1}, "
                      f"records {offset} to {offset + len(batch_texts)}")
            except Exception as e:
                print(f"Error inserting batch: {e}")
                continue
            
        # Flush the new collection
        new_collection.flush()
        
        # Print migration statistics
        print(f"\nMigration completed:")
        print(f"Original collection entities: {total_entities}")
        print(f"New collection entities: {new_collection.num_entities}")
        
        # Release collections
        old_collection.release()
        
    except Exception as e:
        print(f"Migration failed: {e}")
        raise

def main():
    # Connect to Milvus
    connect_to_milvus()

    old_collection_name = "TitalMulti_1"
    new_collection_name = "TitalMulti_1_new"
    
    # Create new collection with correct dimension
    create_collection(new_collection_name, dim=4096)
    
    # Migrate data
    print("\nStarting data migration...")
    migrate_collection_data(old_collection_name, new_collection_name)
    
    # Test search on new collection
    print("\nTesting search on migrated collection...")
    query_text = """i used to go with natalie , but when she and ben got serious , i declined her invitations to spend the holidays with her family . it felt weird taggingalong and i wanted to give them some space . natalie always begged me to join them , but i lied and told her i would be fine and already had plans here . with a pangin my chest , i remembered the first time i spent christmas alone ."""
    
    # Search new collection
    search_collection(new_collection_name, query_text, limit=3)
    
    print("\nMigration and testing completed successfully!")
    print(f"You can now use the new collection '{new_collection_name}'")
    print(f"Once verified, you can drop the old collection '{old_collection_name}' if needed.")

    # Close connection
    connections.disconnect("default")

if __name__ == "__main__":
    main()