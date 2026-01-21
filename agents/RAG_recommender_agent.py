"""
RAG Agent for Warranty Claim Recommendations
Retrieves relevant product policy documents and generates structured review packets
"""

from configparser import ConfigParser
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
from dotenv import load_dotenv
load_dotenv()

config=ConfigParser()
config.read('../model_config.ini')

@dataclass
class HumanReviewPacket:
    """
    A structured packet for human review of AI-verified warranty claims.
    Designed to capture claim validation reasoning, decisions, and follow-up actions.
    """

    # Core evaluation fields
    
    policy_doc_selected: str
    retrieved_policy_docs: List[str] 
    claim_validity: str                     # e.g., "Valid", "Invalid", "Uncertain"
    warranty_coverage: str                  # e.g., "Covered", "Not Covered", "Partially Covered"
    decision: str                           # e.g., "Approve Claim", "Reject Claim", "Escalate for Further Review"

    # Supporting explanation and context
    confidence_score: float
    reasons: List[str] = field(default_factory=list)  # Key justifications or evidence points
    next_steps: List[str] = field(default_factory=list)  # Follow-up actions or recommended process

    # Optional metadata and traceability
    
    claim_id: Optional[str] = None          # Unique claim identifier
    
    notes: Optional[str] = None             # Freeform commentary for contextual clarification
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            
            "policy_doc_selected": self.policy_doc_selected,
            "retrieved_policy_docs": self.retrieved_policy_docs,
            "claim_validity": self.claim_validity,
            "warranty_coverage": self.warranty_coverage,
            "decision": self.decision,
            "confidence_score": self.confidence_score,
            "reasons": self.reasons,
            "next_steps": self.next_steps,
            "claim_id": self.claim_id,
            "notes": self.notes
        }

# State for the RAG workflow
class RAGState(TypedDict):
    claim_info: dict
    policy_doc_selected: str
    retrieved_docs: list
    policy_analysis: str
    recommendation: dict
    review_packet: dict

# Structured output for policy analysis
class PolicyAnalysis(BaseModel):
    is_within_warranty_period: bool = Field(description="Whether claim is within warranty timeframe")
    issue_covered: bool = Field(description="Whether the reported issue is covered by warranty")
    evidence_sufficient: bool = Field(description="Whether provided evidence meets requirements")
    exclusions_apply: bool = Field(description="Whether any warranty exclusions apply")
    analysis_summary: str = Field(description="Brief summary of policy analysis")

# Structured output for recommendation
class ClaimRecommendation(BaseModel):
    claim_validity: Literal["Valid", "Invalid", "Uncertain"] = Field(description="Overall claim validity")
    warranty_coverage: Literal["Covered", "Not Covered", "Partially Covered"] = Field(description="Coverage status")
    decision: Literal["Approve Claim", "Reject Claim", "Escalate for Further Review"] = Field(description="Recommended decision")
    confidence_score: float = Field(description="Confidence in recommendation (0-1)")
    reasons: List[str] = Field(description="Key justifications for the decision")
    next_steps: List[str] = Field(description="Recommended follow-up actions")
    notes: str = Field(description="Additional contextual information")


class RAGRecommendationAgent:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        """
        Initialize the RAG recommendation agent
        
        Args:
            groq_api_key: Your Groq API key
            model: Groq model to use
        """
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
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
        def retrieve_documents(query: str, policy_doc_path: str = "") -> str:
            """
            Retrieve relevant product policy documents based on the claim description.
            
            Args:
                query: The issue description or claim details to search for
                product_model: The product model to filter policies
                
            Returns:
                Concatenated relevant policy documents
            """
            
            filters = {"source": {"$eq": policy_doc_path}} if policy_doc_path != "" else None
            results = self.vector_db.query(query, top_k=5, filters=filters)
            
            docs = [doc.page_content for doc in results]
            return "\n\n".join([f"Policy {i+1}: {doc}" for i, doc in enumerate(docs)])
        
        return [retrieve_documents]
    
    def retrieve_policies(self, state: RAGState):
        """Retrieve relevant policy documents using tools"""
        
        claim_info = state["claim_info"]
        category= claim_info.get("category", "warranty")
        if category != "warranty":
            return {"policy_doc_selected": "N/A", "retrieved_docs": ["Not a warranty claim"]}
        # Extract information from claim_info dictionary
        issue_description = claim_info.get("issue_description", claim_info.get("Issue Description", ""))
        product_model = claim_info.get("product_model", claim_info.get("Product Model", "unknown"))
        
        # If claim_info contains processed_output, extract from there
        if not issue_description and "processed_output" in claim_info:
            issue_description = claim_info.get("processed_output", "")
        directory=config["POLICY_MANUALS"]["directory"]
        
        #Find file in directory matching product model
        product_doc = "unknown"
        for file in os.listdir(directory):
            if product_model in file:
                product_doc = file
                break
        
        # Use tool to retrieve documents
        messages = [
            SystemMessage(content="You are a policy document retrieval assistant. Use the retrieve_documents tool to find relevant warranty policies."),
            HumanMessage(content=f"Retrieve relevant info for the Product Model: {product_model}\nIssue: {issue_description}")
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
        policy_doc_path=os.path.join(directory, product_doc)
        return {"policy_doc_selected": policy_doc_path, "retrieved_docs": retrieved_docs}
    
    def analyze_against_policy(self, state: RAGState):
        """Analyze claim against retrieved policy documents"""
        
        claim_info = state["claim_info"]
        retrieved_docs = state["retrieved_docs"]
        
        analysis_prompt = """You are a warranty policy analyst. Analyze the claim against the provided warranty policies.

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
        
        return {"policy_analysis": analysis.model_dump()}
    
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
        
        return {"recommendation": recommendation.model_dump()}
    
    def create_review_packet(self, state: RAGState):
        """Create structured human review packet"""
        
        recommendation = state["recommendation"]
        
        
        # Create review packet
        packet = HumanReviewPacket(
            
            policy_doc_selected=state["policy_doc_selected"],
            retrieved_policy_docs=state["retrieved_docs"],
            claim_validity=recommendation["claim_validity"],
            warranty_coverage=recommendation["warranty_coverage"],
            decision=recommendation["decision"],
            confidence_score=recommendation["confidence_score"],
            reasons=recommendation["reasons"],
            next_steps=recommendation["next_steps"],
            claim_id=f"CLAIM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            notes=recommendation["notes"]
        )
        
        return {"review_packet": packet.to_dict()}
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retrieve_policies", self.retrieve_policies)
        builder.add_node("analyze_against_policy", self.analyze_against_policy)
        builder.add_node("generate_recommendation", self.generate_recommendation)
        builder.add_node("create_review_packet", self.create_review_packet)
        
        # Add edges
        builder.add_edge(START, "retrieve_policies")
        builder.add_edge("retrieve_policies", "analyze_against_policy")
        builder.add_edge("analyze_against_policy", "generate_recommendation")
        builder.add_edge("generate_recommendation", "create_review_packet")
        builder.add_edge("create_review_packet", END)
        
        return builder.compile()
    
    def process_claim(self, claim_info: dict) -> HumanReviewPacket:
        """
        Process a warranty claim and generate review packet
        
        Args:
            claim_info: Structured claim information from email triage agent
                       Should contain: processed_output, category, confidence, etc.
            
        Returns:
            HumanReviewPacket with recommendation
        """
        state = self.workflow.invoke({
            "claim_info": claim_info
        })
        
        # Convert dict back to HumanReviewPacket
        packet_dict = state["review_packet"]
        return HumanReviewPacket(**packet_dict)


# Example usage
if __name__ == "__main__":
    
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key-here")
    
    # Initialize RAG agent
    rag_agent = RAGRecommendationAgent()
    
    # Mock email processing result from triage agent
    mock_claim_info = {
        "category": "warranty",
        "confidence": 0.95,
        "reasoning": "Email contains warranty claim with damage evidence",
        "processed_output": """WARRANTY CLAIM SUMMARY
======================================================================
From: emily.james@gmail.com
Date: 2026-06-01

Customer Name: John Doe
Contact Info: emily.james@gmail.com, +1-555-123-4567
Product Model: BLD-150
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
    
    # Process claim directly with JSON input
    print("RAG RECOMMENDATION AGENT - PROCESSING CLAIM")
    print("="*70 + "\n")
    
    review_packet = rag_agent.process_claim(mock_claim_info)
    
    print("HUMAN REVIEW PACKET")
    print("="*70)
    print(json.dumps(review_packet.to_dict(), indent=2))
    # print(f"Claim ID: {review_packet.claim_id}")
    # print(f"Claim Details: {review_packet.claim_details}")
    # print(f"Timestamp: {review_packet.review_timestamp}")
    # print(f"\nClaim Validity: {review_packet.claim_validity}")
    # print(f"Warranty Coverage: {review_packet.warranty_coverage}")
    # print(f"Decision: {review_packet.decision}")
    # print(f"\nReasons:")
    # for reason in review_packet.reasons:
    #     print(f"  - {reason}")
    # print(f"\nNext Steps:")
    # for step in review_packet.next_steps:
    #     print(f"  - {step}")
    # print(f"\nNotes: {review_packet.notes}")
    # print("="*70)