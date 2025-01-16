from pymilvus import connections, Collection, utility
import numpy as np
from tabulate import tabulate

def connect_to_milvus(host='localhost', port='19530'):
    """
    Establish connection to Milvus server
    """
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

def get_collection_stats():
    """
    Get statistics for all collections including number of entities
    Returns a list of dictionaries containing collection info
    """
    stats = []
    collections = utility.list_collections()

    for collection_name in collections:
        try:
            collection = Collection(collection_name)
            collection.load()

            # Get collection statistics
            num_entities = collection.num_entities
            schema = collection.schema
            dim = None

            # Try to find vector field and its dimension
            for field in schema.fields:
                if field.dtype == 101:  # DataType.FLOAT_VECTOR
                    dim = field.params.get('dim')
                    break

            stats.append({
                'Collection': collection_name,
                'Entities': num_entities,
                'Dimension': dim,
                'Description': collection.description
            })

            # Release collection from memory
            collection.release()

        except Exception as e:
            print(f"Error getting stats for collection {collection_name}: {e}")
            stats.append({
                'Collection': collection_name,
                'Entities': 'ERROR',
                'Dimension': 'ERROR',
                'Description': str(e)
            })

    return stats

def vector_search(collection, search_vectors, top_k=5, output_fields=None):
    """
    Perform vector similarity search

    Args:
        collection: Milvus collection object
        search_vectors: List of vectors to search for
        top_k: Number of nearest neighbors to retrieve
        output_fields: List of fields to return in the result
    """
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    try:
        results = collection.search(
            data=search_vectors,
            anns_field="embedding",  # Replace with your vector field name
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        raise

def main():
    # Connect to Milvus
    connect_to_milvus()

    # Get statistics for all collections
    stats = get_collection_stats()

    # Print statistics in a nice table format
    if stats:
        print("\nCollection Statistics:")
        print(tabulate(stats, headers='keys', tablefmt='grid'))
    else:
        print("No collections found")

    # Close connection
    connections.disconnect("default")

if __name__ == "__main__":
    main()
