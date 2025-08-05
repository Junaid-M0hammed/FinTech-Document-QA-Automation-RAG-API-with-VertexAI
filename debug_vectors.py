#!/usr/bin/env python3
"""
Debug script to test vector search and similarity scores
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

def debug_vector_search():
    """Debug vector search to see actual similarity scores"""
    load_dotenv()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('fintech-documents')
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"📊 Index Stats:")
    print(f"   Total vectors: {stats.total_vector_count}")
    print(f"   Namespaces: {stats.namespaces}")
    
    if stats.total_vector_count == 0:
        print("❌ No vectors found in the index!")
        return
    
    # Test different queries
    test_queries = [
        "account balance",
        "Who is the account holder?",
        "John Smith",
        "balance",
        "current balance",
        "account"
    ]
    
    print(f"\n🔍 Testing queries with no similarity threshold:")
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Search without filter
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        if results['matches']:
            for i, match in enumerate(results['matches'], 1):
                score = match['score']
                text = match['metadata'].get('text', 'No text')[:100] + "..."
                print(f"   {i}. Score: {score:.3f} | Text: {text}")
        else:
            print("   No matches found")

if __name__ == "__main__":
    debug_vector_search() 