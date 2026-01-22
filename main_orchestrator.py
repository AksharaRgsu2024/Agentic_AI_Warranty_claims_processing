"""
Main Orchestrator for Email Triage and Warranty Claim Processing
Complete LangGraph workflow with Claims Processing, RAG, Human-in-Loop, and Response Generation
"""

from claims_processing_agent import EmailTriageAgent
from RAG_recommender_agent import RAGRecommendationAgent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.graph import StateGraph, START, END
from typing import List, Optional
from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import json
import time
from datetime import datetime
from Mailbox_mockup import MockMailboxAdapter
import os
import re
import logging
from dotenv import load_dotenv
from configparser import ConfigParser
config=ConfigParser()
config.read('model_config.ini')

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- Decision Recording Functions ---

def record_triage_decision(email: dict, triage_result: dict, confidence: float) -> dict:
    """
    Record the triage decision for an email
    
    Args:
        email: Original customer email
        triage_result: Triage results from EmailTriageAgent
        confidence: Confidence score of the triage
        
    Returns:
        Dictionary with recorded triage decision
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "email_from": email.get("from", "N/A"),
        "email_subject": email.get("subject", "N/A"),
        "email_date": email.get("date", "N/A"),
        "triage_category": triage_result.get("category", "unknown"),
        "triage_confidence": confidence,
        "product_model": triage_result.get("product_model", "N/A"),
        "issue_description": triage_result.get("issue_description", "N/A")[:200],  # Truncate for readability
        "reasoning": triage_result.get("reasoning", "N/A")
    }
    return record


def record_claim_decision(email: dict, triage_result: dict, review_packet: dict, human_response: str = None, status: str = "auto") -> dict:
    """
    Record the claim approval/rejection decision
    
    Args:
        email: Original customer email
        triage_result: Triage results
        review_packet: Review packet from RAG agent
        human_response: Human reviewer's decision (if applicable)
        status: "auto" for AI decision, "human" for human override
        
    Returns:
        Dictionary with recorded claim decision
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "decision_status": status,
        "email_from": email.get("from", "N/A"),
        "email_subject": email.get("subject", "N/A"),
        "product_model": triage_result.get("product_model", "N/A"),
        "claim_id": review_packet.get("claim_id", "N/A"),
        "triage_category": triage_result.get("category", "N/A"),
        "claim_validity": review_packet.get("claim_validity", "N/A"),
        "warranty_coverage": review_packet.get("warranty_coverage", "N/A"),
        "ai_decision": review_packet.get("decision", "N/A"),
        "confidence_score": review_packet.get("confidence_score", 0),
        "human_override": human_response if status == "human" else None,
        "reasons_count": len(review_packet.get("reasons", [])),
        "reasons": review_packet.get("reasons", [])[:2],  # First 2 reasons for brevity
        "notes": review_packet.get("notes", "")[:150]  # Truncate for readability
    }
    return record


def save_decisions_to_file(decisions: List[dict], filename: str = "decision_log.json") -> None:
    """
    Save recorded decisions to a JSON file for analysis
    
    Args:
        decisions: List of decision records
        filename: Output filename
    """
    os.makedirs("decision_logs", exist_ok=True)
    filepath = os.path.join("decision_logs", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)
    
    logging.info(f"✅ Decisions saved to {filepath}")


def generate_decision_summary(triage_decisions: List[dict], claim_decisions: List[dict]) -> dict:
    """
    Generate a summary of all decisions made
    
    Args:
        decisions: List of decision records
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_emails_processed": len(triage_decisions),
        "triage_breakdown": {
            "spam": 0,
            "warranty": 0,
            "non_warranty": 0
        },
        "claim_decisions": {
            "approved": 0,
            "rejected": 0,
            "escalated": 0,
            "not_applicable": 0
        },
        "confidence_stats": {
            "average_confidence": 0.0,
            "highest_confidence": 0.0,
            "lowest_confidence": 1.0
        },
        "human_overrides": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    if not triage_decisions:
        return summary
    
    confidence_scores = []
    
    for decision in triage_decisions:
        # Count triage categories
        category = decision.get("triage_category", "unknown")
        if category in summary["triage_breakdown"]:
            summary["triage_breakdown"][category] += 1

    for decision in claim_decisions:    
        # Count claim decisions
        ai_decision = decision.get("ai_decision", "").lower()
        if "approve" in ai_decision:
            summary["claim_decisions"]["approved"] += 1
        elif "reject" in ai_decision:
            summary["claim_decisions"]["rejected"] += 1
        elif "escalate" in ai_decision:
            summary["claim_decisions"]["escalated"] += 1
        else:
            summary["claim_decisions"]["not_applicable"] += 1
        
        # Track confidence scores
        confidence = decision.get("confidence_score", 0)
        if confidence > 0:
            confidence_scores.append(confidence)
            summary["confidence_stats"]["highest_confidence"] = max(
                summary["confidence_stats"]["highest_confidence"], confidence
            )
            summary["confidence_stats"]["lowest_confidence"] = min(
                summary["confidence_stats"]["lowest_confidence"], confidence
            )
        
        # Count human overrides
        if decision.get("decision_status") == "human":
            summary["human_overrides"] += 1
    
    # Calculate average confidence
    if confidence_scores:
        summary["confidence_stats"]["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    return summary


class EvidenceChecklist(TypedDict):
    ModelNumber: str
    SerialNumber: str
    attachments_analysis: str
    validity: str

# Combined state for the entire pipeline
class PipelineState(TypedDict, total=False):
    customer_email: dict
    triage_category: str
    is_warranty: bool
    email_info: dict
    attachment_info: str
    processed_output: str
    product_model: str
    model_code: str
    claims_process_confidence: float
    recommendation: dict
    review_packet: dict
    reviewer_id: Optional[str]
    review_timestamp: Optional[str]
    human_response: str
    human_feedback: str
    final_action: str
    response_email: str
    response_email_json: dict


def claim_processing_agent(state: PipelineState) -> PipelineState:
    """Process incoming email through triage agent"""
    logging.info("📧 Claim email processing node running...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    vlm_model=config["MODEL_CONFIG"]["vlm_model"]
    input_folder=config["EMAILS"]["customer_mails"]
    agent = EmailTriageAgent(
        groq_api_key=groq_api_key,
        attachments_folder=input_folder, model=vlm_model
    )
    
    processed_info = agent.process_email(state["customer_email"])
    
    print("\n" + "="*80)
    print("CLAIMS PROCESSING AGENT OUTPUT")
    print("="*80)
    for item, value in processed_info.items():
        if item != "processed_output":  # Don't print full output here
            print(f"{item}: {value}")
    print("="*80 + "\n")
    
    category = processed_info["category"]
    state["triage_category"] = category
    state["is_warranty"] = (category == "warranty")
    state["claims_process_confidence"] = processed_info["confidence"]
    state["email_info"] = processed_info
    state["processed_output"] = processed_info["processed_output"]
    state["attachment_info"]=processed_info["attachment_analysis"]
    
    # Extract product model from processed_output if not explicitly set
    product_model = "Unknown"
    processed_output = processed_info.get("processed_output", "")
    import re
    model_patterns = [
        r"(?:Product\s+)?Model[:\s]+([A-Z]+-\d+(?:\s+[A-Za-z0-9-]*)?)",
        r"Model[:\s]+([A-Z]+-[\d]+)",
        r"(\b[A-Z]{3}-\d+\b)",
    ]
    for pattern in model_patterns:
        try:
            match = re.search(pattern, processed_output, re.IGNORECASE)
            if match:
                product_model = match.group(1).strip()
                logging.info(f"Extracted product model from triage: {product_model}")
                break
        except re.error:
            continue
    
    state["product_model"] = product_model
    state["model_code"] = product_model.split()[0] if product_model != "Unknown" else "unknown"
    
    return state


def route_decision(state: PipelineState) -> Literal["display_spam", "rag_recommendation_agent", "display_non_warranty"]:
    """Conditional edge function to route based on category"""
    category = state["triage_category"]
    
    if category == "spam":
        return "display_spam"
    elif category == "warranty":
        return "rag_recommendation_agent"
    elif category == "non_warranty":
        return "display_non_warranty"
    
    return "display_spam"  # Default fallback


def rag_recommendation_agent(state: PipelineState) -> PipelineState:
    """Generate warranty claim recommendation using RAG"""
    logging.info("🔍 RAG recommendation node running...")
    
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        rag_model=config["MODEL_CONFIG"]["rag_model"]
        rag_agent = RAGRecommendationAgent(groq_api_key=groq_api_key, model=rag_model)
        
        # Pass the email_info dict to RAG agent
        print("\nPassing to RAG agent:")
        print(f"  Category: {state['email_info'].get('category')}")
        print(f"  Product Model: {state.get('product_model', 'Unknown')}")
        
        result = rag_agent.process_claim(state["email_info"])
        logging.info(f"RAG agent result keys: {result.keys()}")
        
        # Extract the HumanReviewPacket object from the result dict
        if "review_packet" not in result:
            logging.error(f"RAG agent result missing 'review_packet' key. Keys: {result.keys()}")
            raise ValueError(f"RAG agent did not return review_packet. Got keys: {result.keys()}")
        
        review_packet_obj = result["review_packet"]
        logging.info(f"Review packet type: {type(review_packet_obj)}")
        logging.info(f"Review packet decision: {review_packet_obj.decision}")
        logging.info(f"Review packet confidence: {review_packet_obj.confidence_score}")
        
        state["review_packet"] = review_packet_obj.to_dict()
        state["reviewer_id"] = "AI-Agent"
        state["review_timestamp"] = datetime.now().isoformat()
        
        # Evidence info
        email_info = state["email_info"]
        # Use the product_model from state which was extracted in claim_processing_agent
        product_model_display = state.get("product_model", "Unknown")
        
        evidence_info: EvidenceChecklist = {
            "SerialNumber": email_info.get("serial_number", email_info.get("Serial Number", "N/A")),
            "ModelNumber": product_model_display,
            "attachments_analysis": state.get("attachment_info", email_info.get("attachment_analysis", "N/A")),
            "validity": email_info.get("claim_validation", email_info.get("Claim Validation", "Unknown")),
        }

        
        print("\n" + "="*80)
        print("RAG RECOMMENDATION AGENT OUTPUT")
        print("="*80)
        print(f"Claim ID: {review_packet_obj.claim_id}")
        print(f"Product Model: {evidence_info.get('ModelNumber', 'Unknown')}")
        print(f"Claim Validity: {review_packet_obj.claim_validity}")
        print(f"Warranty Coverage: {review_packet_obj.warranty_coverage}")
        print(f"Decision: {review_packet_obj.decision}")
        print(f"Confidence Score: {review_packet_obj.confidence_score:.2f}")
        print(f"Policy Document: {state.get('policy_doc_selected', 'N/A')}")
        print(f"\nEvidence Info:")
        print(f"  Model Number: {evidence_info.get('ModelNumber', 'N/A')}")
        print(f"  Serial Number: {evidence_info.get('SerialNumber', 'N/A')}")
        attachments_text = evidence_info.get('attachments_analysis', 'N/A')
        if attachments_text and attachments_text != 'N/A':
            # Display full attachment analysis if available
            print(f"  Attachments Analysis:")
            for line in attachments_text.split('\n')[:10]:  # Show first 10 lines
                print(f"    {line}")
        else:
            print(f"  Attachments: N/A")
        print(f"  Validity Assessment: {evidence_info.get('validity', 'N/A')}")
        print(f"\nReasons ({len(review_packet_obj.reasons)}):")
        for i, reason in enumerate(review_packet_obj.reasons, 1):
            print(f"  {i}. {reason}")
        print(f"\nNext Steps ({len(review_packet_obj.next_steps)}):")
        for i, step in enumerate(review_packet_obj.next_steps, 1):
            print(f"  {i}. {step}")
        print("="*80 + "\n")
        
        return state
        
    except Exception as e:
        logging.error(f"Error in RAG recommendation agent: {e}", exc_info=True)
        print(f"\n❌ Error in RAG processing: {e}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\nCreating fallback recommendation...\n")
        
        # Create fallback review packet
        fallback_packet = {
            "claim_id": f"CLAIM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "claim_validity": "Uncertain",
            "warranty_coverage": "Not Covered",
            "decision": "Escalate for Further Review",
            "confidence_score": 0.5,
            "evidence_info": {
                "ModelNumber": state.get("product_model", "Unknown"),
                "SerialNumber": "N/A",
                "attachments_analysis": "Error processing",
                "validity": "Requires manual review"
            },
            "reasons": [
                "Automated processing encountered technical issues",
                "Manual review required for accurate assessment"
            ],
            "next_steps": [
                "Forward to warranty specialist for manual review",
                "Contact customer within 24 hours"
            ],
            "policy_doc_selected": "N/A",
            "retrieved_policy_docs": [],
            "notes": f"Error during automated processing: {str(e)[:100]}"
        }
        
        state["review_packet"] = fallback_packet
        state["reviewer_id"] = "AI-Agent (Fallback)"
        state["review_timestamp"] = datetime.now().isoformat()
        
        return state


def display_spam(state: PipelineState) -> PipelineState:
    """Display spam email notification"""
    print("\n" + "="*80)
    print("🚫 SPAM EMAIL DETECTED")
    print("="*80)
    print("Action: Ignored and filtered")
    print(f"From: {state['customer_email'].get('from', 'N/A')}")
    print(f"Subject: {state['customer_email'].get('subject', 'N/A')}")
    print("="*80 + "\n")
    
    state["final_action"] = "spam_filtered"
    return state


def display_non_warranty(state: PipelineState) -> PipelineState:
    """Display non-warranty inquiry"""
    print("\n" + "="*80)
    print("💬 NON-WARRANTY INQUIRY")
    print("="*80)
    print(f"From: {state['customer_email'].get('from', 'N/A')}")
    print(f"Subject: {state['customer_email'].get('subject', 'N/A')}")
    print("\nProcessed Output:")
    print("-"*80)
    print(state["processed_output"])
    print("="*80 + "\n")
    
    state["final_action"] = "non_warranty_displayed"
    return state


def get_human_feedback(state: PipelineState) -> PipelineState:
    """Collect feedback from human reviewer"""
    print("\n" + "="*80)
    print("👋 HUMAN REVIEW REQUIRED")
    print("="*80)
    
    review_packet = state["review_packet"]
    
    print("\nCLAIM DETAILS:")
    print("-"*80)
    # Show first 800 characters of processed output
    output_preview = state["processed_output"][:800]
    if len(state["processed_output"]) > 800:
        output_preview += "..."
    print(output_preview)
    
    print("\n" + "-"*80)
    print("EVIDENCE VERIFICATION:")
    print("-"*80)
    evidence = review_packet.get('evidence_info', {})
    print(f"Product Model: {evidence.get('ModelNumber', 'N/A')}")
    print(f"Serial Number: {evidence.get('SerialNumber', 'N/A')}")
    print(f"Attachments Analysis: {evidence.get('attachments_analysis', 'N/A')[:100]}...")
    print(f"Validity Assessment: {evidence.get('validity', 'N/A')[:100]}...")
    
    print("\n" + "-"*80)
    print("AI RECOMMENDATION:")
    print("-"*80)
    print(f"Claim ID: {review_packet.get('claim_id', 'N/A')}")
    print(f"Product: {state.get('product_model', 'Unknown')}")
    print(f"Claim Validity: {review_packet.get('claim_validity', 'N/A')}")
    print(f"Warranty Coverage: {review_packet.get('warranty_coverage', 'N/A')}")
    print(f"Decision: {review_packet.get('decision', 'N/A')}")
    print(f"Confidence: {review_packet.get('confidence_score', 0):.2%}")
    
    print(f"\nReasons:")
    for i, reason in enumerate(review_packet.get('reasons', []), 1):
        print(f"  {i}. {reason}")
    
    print(f"\nNext Steps:")
    for i, step in enumerate(review_packet.get('next_steps', []), 1):
        print(f"  {i}. {step}")
    
    notes = review_packet.get('notes', 'N/A')
    print(f"\nNotes: {notes[:200]}..." if len(notes) > 200 else f"\nNotes: {notes}")
    print("="*80)
    
    # Get human input
    print("\nOPTIONS:")
    print("  1. approve  - Accept AI recommendation")
    print("  2. reject   - Override and reject claim")
    print("  3. revise   - Re-run workflow with changes")
    
    while True:
        human_decision = input("\nYour decision (approve/reject/revise): ").strip().lower()
        if human_decision in ["approve", "reject", "revise"]:
            break
        print("Invalid input. Please enter 'approve', 'reject', or 'revise'")
    
    feedback = input("Enter any comments or feedback (press Enter to skip): ").strip()
    reviewer_name = input("Enter your name/ID: ").strip() or "Anonymous Reviewer"
    
    print(f"\n✓ Human decision recorded: {human_decision.upper()}")
    print(f"  Reviewer: {reviewer_name}\n")
    
    state["human_response"] = human_decision
    state["human_feedback"] = feedback
    state["reviewer_id"] = reviewer_name
    
    return state


def final_action_route(state: PipelineState) -> Literal["draft_response", "rerun_workflow", "abort"]:
    """Route based on human decision"""
    if state["human_response"] == "approve":
        return "draft_response"
    elif state["human_response"] == "revise":
        return "rerun_workflow"
    else:
        return "abort"


def draft_response(state: PipelineState) -> PipelineState:
    """Draft response email based on approved decision"""
    logging.info("✉️  Drafting response email...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm=config["MODEL_CONFIG"]["response_drafter_model"]
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name=llm
    )
    
    claims_info = state["processed_output"]
    review_packet = state["review_packet"]
    customer_email = state["customer_email"]
    
    # Extract customer name from email
    customer_name = "Valued Customer"
    body_text = customer_email.get("body", "")
    
    # Try to extract name from signature
    name_patterns = [
        r"Sincerely,\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Best regards,\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Thanks,\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, body_text)
        if match:
            customer_name = match.group(1)
            break
    
    # Determine email type based on decision
    decision = review_packet.get("decision", "")
    
    if "Approve" in decision:
        prompt_type = "approval"
        prompt = f"""You are a professional customer service representative for a hairdryer company.

Draft a warm, empathetic email APPROVING a warranty claim.

Customer Name: {customer_name}
Product: {state.get('product_model', 'your hairdryer')}
Claim ID: {review_packet.get('claim_id', 'N/A')}

REQUIREMENTS:
1. Warm greeting addressing customer by name
2. Acknowledge their issue and express concern for the safety/inconvenience
3. Clear statement that warranty claim is APPROVED
4. Explain next steps for replacement/refund process
5. Provide timeline (e.g., "within 3-5 business days")
6. Include claim reference number
7. Offer contact information for questions
8. Professional, empathetic closing

TONE: Professional, warm, reassuring, apologetic for the inconvenience

KEY INFORMATION:
Reasons for approval: {', '.join(review_packet.get('reasons', []))}
Next steps: {', '.join(review_packet.get('next_steps', []))}

Write the complete email body (do not include subject line or from/to fields)."""

    elif "Reject" in decision:
        prompt_type = "rejection"
        prompt = f"""You are a professional customer service representative for a hairdryer company.

Draft a tactful, empathetic email REJECTING a warranty claim.

Customer Name: {customer_name}
Product: {state.get('product_model', 'your hairdryer')}
Claim ID: {review_packet.get('claim_id', 'N/A')}

REQUIREMENTS:
1. Warm greeting addressing customer by name
2. Thank them for reaching out
3. Acknowledge their concern
4. Clearly but tactfully explain why claim cannot be approved
5. Reference specific warranty policy terms
6. Suggest alternative solutions if applicable (repair service, discount on new purchase)
7. Offer to answer questions
8. Maintain goodwill despite rejection
9. Professional closing

TONE: Professional, empathetic, understanding but clear and firm

KEY INFORMATION:
Reasons for rejection: {', '.join(review_packet.get('reasons', []))}
Policy notes: {review_packet.get('notes', '')}

Write the complete email body (do not include subject line or from/to fields)."""

    else:  # Escalation
        prompt_type = "escalation"
        prompt = f"""You are a professional customer service representative for a hairdryer company.

Draft a reassuring email acknowledging a warranty claim that requires FURTHER REVIEW.

Customer Name: {customer_name}
Product: {state.get('product_model', 'your hairdryer')}
Claim ID: {review_packet.get('claim_id', 'N/A')}

REQUIREMENTS:
1. Warm greeting addressing customer by name
2. Thank them for their warranty claim
3. Acknowledge receipt and validate their concern
4. Explain that claim requires specialist review for accurate assessment
5. Provide realistic timeline (24-48 hours)
6. Assure them their case is being prioritized
7. Include claim reference number for tracking
8. Provide contact information
9. Thank them for patience

TONE: Professional, reassuring, transparent, committed

Write the complete email body (do not include subject line or from/to fields)."""
    
    # Generate email content
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Draft the email now for this {prompt_type}.")
    ])
    
    email_body = response.content
    
    # Create email in standard format
    response_email_data = {
        "subject": f"Re: {customer_email.get('subject', 'Warranty Claim')}",
        "from": "warranty.support@breezelite.com",
        "to": customer_email.get("from", ""),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "body": email_body,
        "in_reply_to": customer_email.get("subject", ""),
        "original_date": customer_email.get("date", ""),
        "claim_id": review_packet.get("claim_id", ""),
        "decision_type": prompt_type,
        "customer_email_reference": customer_email.get("filename", "unknown"),
        "attachments": []
    }
    
    state["response_email"] = email_body
    state["response_email_json"] = response_email_data
    state["final_action"] = "response_drafted"
    
    print("\n" + "="*80)
    print("✉️  RESPONSE EMAIL DRAFTED")
    print("="*80)
    print(f"Type: {prompt_type.upper()}")
    print(f"To: {response_email_data['to']}")
    print(f"Subject: {response_email_data['subject']}")
    print(f"Reference: {response_email_data.get('customer_email_reference', 'unknown')}")
    print("\n" + "-"*80)
    print("EMAIL BODY:")
    print("-"*80)
    print(email_body)
    print("="*80 + "\n")
    
    return state


def save_response_email(state: PipelineState) -> PipelineState:
    """Save response email to JSON file"""
    logging.info("💾 Saving response email...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_emails/response_{timestamp}.json"
    
    os.makedirs("response_emails", exist_ok=True)
    
    response_data = state["response_email_json"]
    
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)
    
    print("="*80)
    print(f"✅ RESPONSE EMAIL SAVED")
    print("="*80)
    print(f"File: {filename}")
    print(f"To: {response_data['to']}")
    print(f"Subject: {response_data['subject']}")
    print(f"Claim ID: {response_data.get('claim_id', 'N/A')}")
    print("="*80 + "\n")
    
    state["final_action"] = "completed"
    return state


def rerun_workflow(state: PipelineState) -> PipelineState:
    """Handle workflow revision request"""
    print("\n" + "="*80)
    print("🔄 WORKFLOW REVISION REQUESTED")
    print("="*80)
    print("Note: In production, this would re-run the RAG agent with updated parameters.")
    print("For now, marking as requiring manual intervention.")
    print("="*80 + "\n")
    
    state["final_action"] = "revision_required"
    return state


def abort(state: PipelineState) -> PipelineState:
    """Handle rejected/aborted claims"""
    print("\n" + "="*80)
    print("❌ CLAIM PROCESSING ABORTED")
    print("="*80)
    print(f"Reviewer: {state.get('reviewer_id', 'N/A')}")
    print(f"Feedback: {state.get('human_feedback', 'None provided')}")
    print("Action: Claim marked for manual review")
    print("="*80 + "\n")
    
    state["final_action"] = "aborted"
    return state


def build_workflow():
    """Build the complete LangGraph workflow"""
    orchestrator = StateGraph(PipelineState)
    
    # Add all nodes
    orchestrator.add_node("claim_processing_agent", claim_processing_agent)
    orchestrator.add_node("rag_recommendation_agent", rag_recommendation_agent)
    orchestrator.add_node("display_spam", display_spam)
    orchestrator.add_node("display_non_warranty", display_non_warranty)
    orchestrator.add_node("get_human_feedback", get_human_feedback)
    orchestrator.add_node("draft_response", draft_response)
    orchestrator.add_node("save_response_email", save_response_email)
    orchestrator.add_node("rerun_workflow", rerun_workflow)
    orchestrator.add_node("abort", abort)
    
    # Build workflow edges
    orchestrator.add_edge(START, "claim_processing_agent")
    
    # Route after triage
    orchestrator.add_conditional_edges(
        "claim_processing_agent",
        route_decision,
        {
            "display_spam": "display_spam",
            "rag_recommendation_agent": "rag_recommendation_agent",
            "display_non_warranty": "display_non_warranty"
        }
    )
    
    # Spam and non-warranty end immediately
    orchestrator.add_edge("display_spam", END)
    orchestrator.add_edge("display_non_warranty", END)
    
    # Warranty claims go through RAG then human review
    orchestrator.add_edge("rag_recommendation_agent", "get_human_feedback")
    
    # Route based on human decision
    orchestrator.add_conditional_edges(
        "get_human_feedback",
        final_action_route,
        {
            "draft_response": "draft_response",
            "rerun_workflow": "rerun_workflow",
            "abort": "abort"
        }
    )
    
    # Draft response leads to save
    orchestrator.add_edge("draft_response", "save_response_email")
    
    # All terminal nodes lead to END
    orchestrator.add_edge("save_response_email", END)
    orchestrator.add_edge("rerun_workflow", END)
    orchestrator.add_edge("abort", END)
    
    return orchestrator.compile()


def process_mailbox(inbox_dir: str = "./inbox"):
    """Process all emails in mailbox through the workflow"""
    
    print("\n" + "="*80)
    print("🚀 EMAIL PROCESSING ORCHESTRATOR - STARTING")
    print("="*80 + "\n")
    
    workflow = build_workflow()
    adapter = MockMailboxAdapter(inbox_dir=inbox_dir, poll_interval=2.0)
    
    results = []
    triage_decisions = []
    claim_decisions = []
    
    for i, email in enumerate(adapter.read_all_once(), 1):
        print(f"\n{'='*80}")
        print(f"📨 PROCESSING EMAIL {i}")
        print(f"{'='*80}\n")
        
        # Run through workflow
        result = workflow.invoke({"customer_email": email})
        results.append(result)
        
        # Record triage decision
        triage_record = record_triage_decision(
            email=email,
            triage_result=result.get("email_info", {}),
            confidence=result.get("claims_process_confidence", 0)
        )
        triage_decisions.append(triage_record)
        
        # Record claim decision (if warranty claim)
        if result.get("triage_category") == "warranty" and result.get("review_packet"):
            claim_record = record_claim_decision(
                email=email,
                triage_result=result.get("email_info", {}),
                review_packet=result.get("review_packet", {}),
                human_response=result.get("human_response"),
                status="human" if result.get("human_response") else "auto"
            )
            claim_decisions.append(claim_record)
        
        print(f"\n✓ Email {i} processing complete")
        print(f"  Category: {result.get('triage_category', 'unknown').upper()}")
        print(f"  Final Action: {result.get('final_action', 'unknown')}\n")
    
    # Save all decisions to files
    print("\n" + "="*80)
    print("📝 RECORDING DECISIONS")
    print("="*80)
    
    # if triage_decisions:
    #     save_decisions_to_file(triage_decisions, f"triage_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    if claim_decisions:
        save_decisions_to_file(claim_decisions, f"claim_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Generate and display summary
    
    summary_stats = generate_decision_summary(triage_decisions, claim_decisions)
    
    print("\n" + "="*80)
    print(f"📊 PROCESSING SUMMARY - {len(results)} emails processed")
    print("="*80)
    
    print(f"\nTriage Breakdown:")
    print(f"  Spam: {summary_stats['triage_breakdown']['spam']}")
    print(f"  Warranty Claims: {summary_stats['triage_breakdown']['warranty']}")
    print(f"  Non-Warranty: {summary_stats['triage_breakdown']['non_warranty']}")
    
    print(f"\nClaim Decisions ({len(claim_decisions)} warranty claims):")
    print(f"  Approved: {summary_stats['claim_decisions']['approved']}")
    print(f"  Rejected: {summary_stats['claim_decisions']['rejected']}")
    print(f"  Escalated: {summary_stats['claim_decisions']['escalated']}")
    
    print(f"\nConfidence Metrics:")
    print(f"  Average: {summary_stats['confidence_stats']['average_confidence']:.2%}")
    print(f"  Highest: {summary_stats['confidence_stats']['highest_confidence']:.2%}")
    print(f"  Lowest: {summary_stats['confidence_stats']['lowest_confidence']:.2%}")
    
    if summary_stats['human_overrides'] > 0:
        print(f"\nHuman Overrides: {summary_stats['human_overrides']}")
    
    # Save summary
    summary_file = f"decision_logs/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("decision_logs", exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Summary saved to {summary_file}")
    
    print("="*80 + "\n")
    
    return results


# Main execution
if __name__ == "__main__":
    inbox_dir=config["EMAILS"]["customer_mails"]
    results = process_mailbox(inbox_dir=inbox_dir)
    print("✅ All emails processed successfully!")