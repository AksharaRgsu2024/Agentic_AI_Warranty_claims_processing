#!/usr/bin/env python3
"""Test full RAG agent flow with real email"""

import json
import logging
import os
from dotenv import load_dotenv

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(funcName)-25s | %(message)s'
)

load_dotenv()

# Test email
email_1_path = "test_customer_emails/email_1.json"
if not os.path.exists(email_1_path):
    print(f"❌ Test email not found at {email_1_path}")
    exit(1)

with open(email_1_path, 'r') as f:
    test_email = json.load(f)

print("="*80)
print("FULL RAG AGENT TEST")
print("="*80)
print(f"Email: {test_email.get('Subject', 'N/A')}")
print(f"From: {test_email.get('From', 'N/A')}")
print()

try:
    # First run the claims processing agent to triage
    print("Step 1: Email Triage")
    print("-"*80)
    from claims_processing_agent import EmailTriageAgent
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not set")
        exit(1)
    
    triage_agent = EmailTriageAgent(groq_api_key=groq_api_key)
    triage_result = triage_agent.process_email(test_email)
    
    print(f"✓ Triage complete")
    print(f"  Category: {triage_result.get('category')}")
    print(f"  Confidence: {triage_result.get('confidence'):.2%}")
    print()
    
    if triage_result.get('category') != 'warranty':
        print("❌ Expected warranty category, got:", triage_result.get('category'))
        exit(1)
    
    # Now run RAG agent
    print("Step 2: RAG Recommendation")
    print("-"*80)
    from RAG_recommender_agent import RAGRecommendationAgent
    
    rag_agent = RAGRecommendationAgent(groq_api_key=groq_api_key, model="mixtral-8x7b-32768")
    print("✓ RAG agent initialized")
    print()
    
    print("Calling process_claim...")
    result = rag_agent.process_claim(triage_result)
    
    print(f"\n✓ RAG agent returned")
    print(f"  Result keys: {list(result.keys())}")
    
    if "review_packet" not in result:
        print(f"❌ Missing review_packet in result. Keys: {result.keys()}")
        exit(1)
    
    review_packet = result["review_packet"]
    print(f"  Review packet type: {type(review_packet)}")
    
    # Display results
    print()
    print("="*80)
    print("REVIEW PACKET RESULTS")
    print("="*80)
    print(f"Claim ID: {review_packet.claim_id}")
    print(f"Claim Validity: {review_packet.claim_validity}")
    print(f"Warranty Coverage: {review_packet.warranty_coverage}")
    print(f"Decision: {review_packet.decision}")
    print(f"Confidence Score: {review_packet.confidence_score:.2%}")
    print(f"\nReasons:")
    for i, reason in enumerate(review_packet.reasons, 1):
        print(f"  {i}. {reason}")
    print(f"\nNext Steps:")
    for i, step in enumerate(review_packet.next_steps, 1):
        print(f"  {i}. {step}")
    print()
    
    # Check if using defaults
    if review_packet.decision == "Escalate for Further Review" and review_packet.confidence_score == 0.5:
        print("⚠️  WARNING: Review packet is using fallback values!")
        print("This suggests the JSON parsing failed.")
    else:
        print("✓ Review packet contains parsed values (not defaults)")
    
    print("="*80)
    print("✓ Test completed successfully")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
