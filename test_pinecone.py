#!/usr/bin/env python3
"""
Simple test script to verify Pinecone credentials and connection
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

def test_pinecone_connection():
    """Test Pinecone connection with current credentials"""
    load_dotenv()
    
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    
    print(f"Testing Pinecone connection...")
    print(f"API Key: {'✅ Found' if api_key else '❌ Missing'}")
    print(f"Environment: {environment}")
    print(f"API Key length: {len(api_key) if api_key else 0} characters")
    print(f"API Key prefix: {api_key[:10] + '...' if api_key else 'None'}")
    
    try:
        print("\n🔄 Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        
        print("✅ Pinecone client created successfully")
        
        print("\n🔄 Testing list_indexes()...")
        indexes = list(pc.list_indexes())
        print(f"✅ Successfully connected! Found {len(indexes)} indexes:")
        
        for index in indexes:
            print(f"  - {index.name} (dimension: {index.dimension}, metric: {index.metric})")
            
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection failed:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n💡 This looks like an authentication issue. Please check:")
            print("   1. Your API key is correct (copy from Pinecone dashboard)")
            print("   2. Your Pinecone account has billing set up")
            print("   3. You're using the right project/organization")
            
        elif "403" in str(e) or "Forbidden" in str(e):
            print("\n💡 This looks like a permissions issue.")
            
        return False

if __name__ == "__main__":
    test_pinecone_connection() 