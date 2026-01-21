"""
RAG Agent for Warranty Claim Recommendations
Retrieves relevant product policy documents and generates structured review packets
"""

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import List, Optional
from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from vector_db import VectorDB
from configparser import ConfigParser
config=ConfigParser()
config.read('model_config.ini')

@dataclass
class HumanReviewPacket:
    """
    A structured packet for human review of AI-verified warranty claims.
    Designed to capture claim validation reasoning, decisions, and follow-up actions.
    """

    # Core evaluation fields
    claim_validity: str                     # e.g., "Valid", "Invalid", "Uncertain"
    warranty_coverage: str                  # e.g., "Covered", "Not Covered", "Partially Covered"
    decision: str                           # e.g., "Approve Claim", "Reject Claim", "Escalate for Further Review"

    # Supporting explanation and context
    reasons: List[str] = field(default_factory=list)  # Key justifications or evidence points
    next_steps: List[str] = field(default_factory=list)  # Follow-up actions or recommended process

    # Optional metadata and traceability
    reviewer_id: Optional[str] = None       # Human reviewer's ID or name
    claim_id: Optional[str] = None          # Unique claim identifier
    review_timestamp: Optional[str] = None  # ISO8601 timestamp of review
    notes: Optional[str] = None             # Freeform commentary for contextual clarification
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "claim_validity": self.claim_validity,
            "warranty_coverage": self.warranty_coverage,
            "decision": self.decision,
            "reasons": self.reasons,
            "next_steps": self.next_steps,
            "reviewer_id": self.reviewer_id,
            "claim_id": self.claim_id,
            "review_timestamp": self.review_timestamp,
            "notes": self.notes
        }

# State for the RAG workflow
class RAGState(TypedDict):
    claim_info: dict
    category: str
    retrieved_docs: list
    policy_analysis: str
    recommendation: dict
    review_packet: dict
    customer_response: str

# Structured output for policy analysis
class PolicyAnalysis(BaseModel):
    is_within_warranty_period: bool = Field(description="Whether claim is within warranty timeframe")
    issue_covered: bool = Field(description="Whether the reported issue is covered by warranty")
    evidence_sufficient: bool = Field(description="Whether provided evidence meets requirements")
    exclusions_apply: bool = Field(description="Whether any warranty exclusions apply")
    analysis_summary: str = Field(description="Brief summary of policy analysis")

# Structured output for recommendation
class ClaimRecommendation(BaseModel):
    claim_validity: Literal["Valid", "Invalid", "Uncertain", "N/A"] = Field(description="Overall claim validity")
    warranty_coverage: Literal["Covered", "Not Covered", "Partially Covered", "N/A"] = Field(description="Coverage status")
    decision: Literal["Approve Claim", "Reject Claim", "Escalate for Further Review", "Ignore", "Provide Information"] = Field(description="Recommended decision")
    confidence_score: float = Field(description="Confidence in recommendation (0-1)")
    reasons: List[str] = Field(description="Key justifications for the decision")
    next_steps: List[str] = Field(description="Recommended follow-up actions")
    notes: str = Field(description="Additional contextual information")
    customer_response: Optional[str] = Field(default=None, description="Direct response to customer for non-warranty inquiries")


class RAGRecommendationAgent:
    def __init__(self, groq_api_key: str, model: str = "openai/gpt-oss-20b"):
        """
        Initialize the RAG recommendation agent
        
        Args:
            groq_api_key: Your Groq API key
            model: Groq model to use
        """
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name=model
        )
        
        # Create structured output LLMs
        self.policy_analyzer = self.llm.with_structured_output(PolicyAnalysis)
        self.recommender = self.llm.with_structured_output(ClaimRecommendation)
        
        self.vector_db = VectorDB(index_name=config.get("VECTOR_DB", "index_name"))
        
        # Create tools
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _create_tools(self):
        """Create tools for document retrieval"""
        
        @tool
        def retrieve_documents(query: str, product_model_doc: str = "unknown") -> str:
            """
            Retrieve relevant product policy documents based on the claim description.
            
            Args:
                query: The issue description or claim details to search for
                product_model_doc: The product model document to filter policies
                
            Returns:
                Concatenated relevant policy documents
            """
            filters = {"source": {"$eq": product_model_doc}} if product_model_doc != "unknown" else None
            results = self.vector_db.query(query, top_k=5, filters=filters)
            
            docs = [doc.page_content for doc in results]
            return "\n\n".join([f"Policy {i+1}: {doc}" for i, doc in enumerate(docs)])
        
        return [retrieve_documents]
    
    def extract_category(self, state: RAGState):
        """Extract email category from claim_info"""
        claim_info = state["claim_info"]
        category = claim_info.get("category", "unknown").lower()
        
        return {"category": category}
    
    def route_by_category(self, state: RAGState):
        """Route based on email category"""
        category = state["category"]
        
        if category == "spam":
            return "handle_spam"
        elif category == "non_warranty":
            return "handle_non_warranty"
        elif category == "warranty":
            return "retrieve_policies"
        else:
            # Default to warranty processing for unknown
            return "retrieve_policies"
    
    def handle_spam(self, state: RAGState):
        """Handle spam emails - ignore with minimal processing"""
        
        packet = HumanReviewPacket(
            claim_validity="N/A",
            warranty_coverage="N/A",
            decision="Ignore",
            reasons=["Email classified as spam"],
            next_steps=["No action required", "Archive/delete email"],
            reviewer_id="AI-Agent",
            claim_id=f"SPAM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            review_timestamp=datetime.now().isoformat(),
            notes="Spam email - no processing needed"
        )
        
        return {"review_packet": packet.to_dict()}
    
    def handle_non_warranty(self, state: RAGState):
        """Handle non-warranty customer inquiries"""
        
        claim_info = state["claim_info"]
        processed_output = claim_info.get("processed_output", "")
        
        # Extract customer question/inquiry
        extraction_prompt = """Extract the main customer question or inquiry from this customer service summary.
Provide just the core question the customer is asking."""

        question_response = self.llm.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=processed_output)
        ])
        
        customer_question = question_response.content
        
        # Retrieve relevant documents
        docs_text = self.tools[0].invoke({
            "query": customer_question,
            "product_model": claim_info.get("product_model", "general")
        })
        
        # Generate answer from retrieved docs
        answer_prompt = """You are a customer service representative for a hairdryer company.
Based on the retrieved product information and policies, answer the customer's question.

If the retrieved documents contain the answer:
- Provide a clear, helpful response
- Reference specific product details
- Be friendly and professional

If the documents don't contain enough information:
- Acknowledge the question
- Indicate that you'll need to escalate to a specialist
- Suggest alternative resources if applicable"""

        answer_response = self.llm.invoke([
            SystemMessage(content=answer_prompt),
            HumanMessage(content=f"""
Customer Question:
{customer_question}

Retrieved Information:
{docs_text}

Please provide a helpful response.""")
        ])
        
        customer_response = answer_response.content
        
        # Determine if we can answer or need to escalate
        can_answer = "I don't have enough information" not in customer_response and "escalate" not in customer_response.lower()
        
        packet = HumanReviewPacket(
            claim_validity="N/A",
            warranty_coverage="N/A",
            decision="Provide Information" if can_answer else "Escalate for Further Review",
            reasons=[
                "Customer inquiry (non-warranty)",
                "Answer generated from product documentation" if can_answer else "Insufficient information in documentation"
            ],
            next_steps=[
                "Send prepared response to customer" if can_answer else "Forward to product specialist",
                "Track customer satisfaction" if can_answer else "Provide detailed answer within 24 hours"
            ],
            reviewer_id="AI-Agent",
            claim_id=f"INQUIRY-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            review_timestamp=datetime.now().isoformat(),
            notes=f"Non-warranty inquiry. {'Answered from documentation.' if can_answer else 'Requires specialist input.'}"
        )
        
        return {
            "review_packet": packet.to_dict(),
            "customer_response": customer_response
        }
    
    def retrieve_policies(self, state: RAGState):
        """Retrieve relevant policy documents using tools"""
        
        claim_info = state["claim_info"]
        
        # Extract information from claim_info dictionary
        issue_description = claim_info.get("issue_description", claim_info.get("Issue Description", ""))
        product_model = claim_info.get("product_model", claim_info.get("Product Model", "unknown"))
        directory=config["POLICY_MANUALS"]["directory"]
        
        #Find file in directory matching product model
        product_doc = "unknown"
        for file in os.listdir(directory):
            if product_model in file:
                product_doc = file
                break
        product_doc_path=os.path.join(directory, product_doc)
        # If claim_info contains processed_output, extract from there
        if not issue_description and "processed_output" in claim_info:
            issue_description = claim_info.get("processed_output", "")
        
        # Use tool to retrieve documents
        messages = [
            SystemMessage(content="You are a policy document retrieval assistant. Use the retrieve_documents tool to find relevant warranty policies."),
            HumanMessage(content=f"Retrieve relevant info from the product document: {product_doc_path}\nIssue: {issue_description}")
        ]
        
        result = self.llm_with_tools.invoke(messages)
        
        # Execute tool if called
        retrieved_docs = []
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tool_call in result.tool_calls:
                if tool_call['name'] == 'retrieve_documents':
                    docs_text = self.tools[0].invoke(tool_call['args'])
                    retrieved_docs.append(docs_text)
        else:
            # Fallback: directly call tool
            docs_text = self.tools[0].invoke({
                "query": issue_description,
                "product_model": product_model
            })
            retrieved_docs.append(docs_text)
        
        return {"policy_doc_selected": product_doc_path,"retrieved_docs": retrieved_docs}
    
    def analyze_against_policy(self, state: RAGState):
        """Analyze claim against retrieved policy documents"""
        
        claim_info = state["claim_info"]
        retrieved_docs = state["retrieved_docs"]
        
        analysis_prompt = """You are a warranty policy analyst. Analyze the claim against the warranty policies.

Check:
1. Is the claim within the warranty period?
2. Is the reported issue covered by warranty?
3. Is the evidence sufficient (valid receipt, damage proof)?
4. Do any exclusions apply?

Provide structured analysis."""

        context = f"""
Claim Information:
{json.dumps(claim_info, indent=2)}

Relevant Warranty Policies:
{chr(10).join(retrieved_docs)}
"""

        analysis = self.policy_analyzer.invoke([
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=context)
        ])
        
        return {"policy_analysis": analysis.dict()}
    
    def generate_recommendation(self, state: RAGState):
        """Generate structured recommendation based on analysis"""
        
        claim_info = state["claim_info"]
        policy_analysis = state["policy_analysis"]
        retrieved_docs = state["retrieved_docs"]
        
        recommendation_prompt = """You are a warranty claim decision specialist. Based on the claim information and policy analysis, provide a structured recommendation.

Consider:
- Policy compliance
- Evidence quality
- Safety implications
- Customer service impact

Provide clear decision with justification and next steps."""

        context = f"""
Claim Information:
{json.dumps(claim_info, indent=2)}

Policy Analysis:
{json.dumps(policy_analysis, indent=2)}

Warranty Policies:
{chr(10).join(retrieved_docs)}
"""

        recommendation = self.recommender.invoke([
            SystemMessage(content=recommendation_prompt),
            HumanMessage(content=context)
        ])
        
        return {"recommendation": recommendation.dict()}
    
    def create_review_packet(self, state: RAGState):
        """Create structured human review packet"""
        
        recommendation = state["recommendation"]
        claim_info = state["claim_info"]
        
        # Create review packet
        packet = HumanReviewPacket(
            claim_validity=recommendation["claim_validity"],
            warranty_coverage=recommendation["warranty_coverage"],
            decision=recommendation["decision"],
            reasons=recommendation["reasons"],
            next_steps=recommendation["next_steps"],
            reviewer_id="AI-Agent",
            claim_id=f"CLAIM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            review_timestamp=datetime.now().isoformat(),
            notes=recommendation["notes"]
        )
        
        return {"review_packet": packet.to_dict()}
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("extract_category", self.extract_category)
        builder.add_node("handle_spam", self.handle_spam)
        builder.add_node("handle_non_warranty", self.handle_non_warranty)
        builder.add_node("retrieve_policies", self.retrieve_policies)
        builder.add_node("analyze_against_policy", self.analyze_against_policy)
        builder.add_node("generate_recommendation", self.generate_recommendation)
        builder.add_node("create_review_packet", self.create_review_packet)
        
        # Add edges
        builder.add_edge(START, "extract_category")
        
        # Conditional routing based on category
        builder.add_conditional_edges(
            "extract_category",
            self.route_by_category,
            {
                "handle_spam": "handle_spam",
                "handle_non_warranty": "handle_non_warranty",
                "retrieve_policies": "retrieve_policies"
            }
        )
        
        # Spam path goes directly to END
        builder.add_edge("handle_spam", END)
        
        # Non-warranty path goes directly to END
        builder.add_edge("handle_non_warranty", END)
        
        # Warranty path continues through analysis
        builder.add_edge("retrieve_policies", "analyze_against_policy")
        builder.add_edge("analyze_against_policy", "generate_recommendation")
        builder.add_edge("generate_recommendation", "create_review_packet")
        builder.add_edge("create_review_packet", END)
        
        return builder.compile()
    
    def process_claim(self, claim_info: dict) -> dict:
        """
        Process an email and generate review packet
        
        Args:
            claim_info: Structured claim information from email triage agent
                       Should contain: category, processed_output, confidence, etc.
            
        Returns:
            Dictionary containing:
            - review_packet: HumanReviewPacket object
            - customer_response: Optional response text for non-warranty inquiries
        """
        state = self.workflow.invoke({
            "claim_info": claim_info
        })
        
        # Convert dict back to HumanReviewPacket
        packet_dict = state["review_packet"]
        packet = HumanReviewPacket(**packet_dict)
        
        result = {
            "review_packet": packet,
            "category": state.get("category", "unknown")
        }
        
        # Include customer response for non-warranty inquiries
        if state.get("customer_response"):
            result["customer_response"] = state["customer_response"]
        
        return result


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key-here")
    
    # Initialize RAG agent
    rag_agent = RAGRecommendationAgent(groq_api_key=GROQ_API_KEY)
    
    print("RAG RECOMMENDATION AGENT - TESTING ALL CATEGORIES")
    print("="*70 + "\n")
    
    # Test 1: Warranty claim
    print("TEST 1: WARRANTY CLAIM")
    print("-"*70)
    warranty_claim = {
        "category": "warranty",
        "confidence": 0.95,
        "reasoning": "Email contains warranty claim with damage evidence",
        "processed_output": """WARRANTY CLAIM SUMMARY
======================================================================
From: emily.james@gmail.com
Date: 2026-06-01

Customer Name: John Doe
Contact Info: emily.james@gmail.com, +1-555-123-4567
Product Model: BreezeLite Everyday Dryer (Model BLD-150)
Issue Description: Hairdryer producing visible sparks and loud popping sounds near safety plug. High severity - safety hazard.
Purchase Date: 2026-03-15 (within 3-month warranty)
Evidence Provided: Receipt showing purchase, images of sparking damage
Claim Validation: Visual evidence confirms sparking near plug, matches customer description
Safety Concerns: Electrical hazard - immediate discontinuation advised
Priority Level: URGENT
Recommended Action: Immediate replacement or refund under warranty""",
        "attachment_analysis": "Receipt verified, damage images show electrical sparking",
        "product_model": "BLD-150",
        "issue_description": "Sparking and popping sounds near safety plug - electrical hazard"
    }
    
    result = rag_agent.process_claim(warranty_claim)
    review_packet = result["review_packet"]
    
    print(f"Category: {result['category']}")
    print(f"Decision: {review_packet.decision}")
    print(f"Validity: {review_packet.claim_validity}")
    print(f"Coverage: {review_packet.warranty_coverage}")
    print("Reasons:", review_packet.reasons[:2])
    print("\n" + "="*70 + "\n")
    
    # Test 2: Non-warranty inquiry
    print("TEST 2: NON-WARRANTY INQUIRY")
    print("-"*70)
    non_warranty_inquiry = {
        "category": "non_warranty",
        "confidence": 0.88,
        "reasoning": "Customer asking about product features",
        "processed_output": """CUSTOMER INQUIRY SUMMARY
======================================================================
From: sarah.customer@email.com
Date: 2026-06-02

Customer details: Sarah Customer, sarah.customer@email.com
Type of inquiry: Product information request
Specific questions asked: How many heat settings does the SuperDry 5000 have? Is it suitable for thick hair?
Customer intent: Pre-purchase research
Suggested response approach: Provide detailed product specifications""",
        "product_model": "SuperDry 5000"
    }
    
    result = rag_agent.process_claim(non_warranty_inquiry)
    review_packet = result["review_packet"]
    
    print(f"Category: {result['category']}")
    print(f"Decision: {review_packet.decision}")
    if "customer_response" in result:
        print(f"Customer Response: {result['customer_response'][:150]}...")
    print("Next Steps:", review_packet.next_steps)
    print("\n" + "="*70 + "\n")
    
    # Test 3: Spam
    print("TEST 3: SPAM EMAIL")
    print("-"*70)
    spam_email = {
        "category": "spam",
        "confidence": 0.99,
        "reasoning": "Promotional email with unrelated content",
        "processed_output": "SPAM - No action required. Email filtered."
    }
    
    result = rag_agent.process_claim(spam_email)
    review_packet = result["review_packet"]
    
    print(f"Category: {result['category']}")
    print(f"Decision: {review_packet.decision}")
    print("Reasons:", review_packet.reasons)
    print("Next Steps:", review_packet.next_steps)
    print("\n" + "="*70)