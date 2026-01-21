#!/usr/bin/env python3
"""Quick test of RAG agent on first email"""

import json
import logging
import os
import sys
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

load_dotenv()

# Load first test email
test_email_path = "test_customer_emails/email_1.json"

if not os.path.exists(test_email_path):
    print(f"❌ Test email not found at {test_email_path}")
    sys.exit(1)

with open(test_email_path, 'r') as f:
    email_data = json.load(f)

print("="*80)
print("Testing RAG Agent with First Email")
print("="*80)
print(f"Subject: {email_data.get('Subject', 'N/A')}")
print(f"From: {email_data.get('From', 'N/A')}")
print("\nInitializing RAG agent...")

try:
    from RAG_recommender_agent import RAGRecommendationAgent
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not set in environment")
        sys.exit(1)
    
    rag_agent = RAGRecommendationAgent(groq_api_key=groq_api_key, model="mixtral-8x7b-32768")
    print("✓ RAG agent initialized")
    
    print("\nProcessing claim...")
    result = rag_agent.process_claim(email_data)
    
    print(f"\nResult keys: {result.keys()}")
    
    if "review_packet" in result:
        review_packet = result["review_packet"]
        print(f"\n✓ Review packet received")
        print(f"  Type: {type(review_packet)}")
        print(f"  Fields: {vars(review_packet) if hasattr(review_packet, '__dict__') else review_packet}")
        
        # Try to convert to dict
        if hasattr(review_packet, 'to_dict'):
            packet_dict = review_packet.to_dict()
            print(f"\n✓ Converted to dict with keys: {packet_dict.keys()}")
            print(f"  Decision: {packet_dict.get('decision', 'N/A')}")
            print(f"  Confidence: {packet_dict.get('confidence_score', 'N/A')}")
        else:
            print(f"❌ Review packet missing to_dict() method")
    else:
        print(f"❌ No review_packet in result")
        print(f"  Available keys: {result.keys()}")
        for key, value in result.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {type(value)}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ Test completed successfully")
print("="*80)
