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
import logging

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
    confidence_score: float = 0.0           # Confidence in recommendation (0-1)
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
            "confidence_score": self.confidence_score,
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
    warranty_coverage: Literal["Covered", "Not Covered", "Partially Covered", "N/A"] = Field(default="N/A", description="Coverage status")
    decision: Literal["Approve Claim", "Reject Claim", "Escalate for Further Review", "Ignore", "Provide Information"] = Field(description="Recommended decision")
    confidence_score: float = Field(description="Confidence in recommendation (0-1)")
    reasons: List[str] = Field(description="Key justifications for the decision")
    next_steps: List[str] = Field(default_factory=list, description="Recommended follow-up actions")
    notes: str = Field(default="", description="Additional contextual information")
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
        
        # Use direct LLM without structured output tools
        # The LLM will generate JSON text responses which we parse manually
        # This avoids Groq's tool validation issues while still getting structured output
        self.policy_analyzer = self.llm
        self.recommender = self.llm
        
        self.vector_db = VectorDB(index_name=config.get("VECTOR_DB", "index_name"))
        
        # Ensure documents are loaded into VectorDB
        policy_dir = config.get("POLICY_MANUALS", "directory")
        try:
            # Check if documents need to be loaded by trying a dummy query
            test_results = self.vector_db.query("test", top_k=1, filters=None)
            if not test_results:
                logging.info(f"VectorDB empty, loading documents from {policy_dir}")
                self.vector_db.process_upsert(policy_dir)
        except Exception as e:
            logging.warning(f"Error checking VectorDB: {e}. Attempting to load documents...")
            try:
                self.vector_db.process_upsert(policy_dir)
            except Exception as e2:
                logging.error(f"Failed to load documents into VectorDB: {e2}")
        
        # Create tools
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _parse_response(self, response, model_class=None):
        """Parse response from LLM - handles both structured output and raw JSON"""
        try:
            import json
            import re
            from langchain_core.messages import BaseMessage
            
            logging.debug(f"_parse_response called with type: {type(response)}, is BaseMessage: {isinstance(response, BaseMessage)}")
            
            # Check for AIMessage/BaseMessage first - these have content that needs parsing
            if isinstance(response, BaseMessage):
                content = response.content
                logging.debug(f"_parse_response: Processing BaseMessage content (length: {len(content)})")
                logging.debug(f"Response content first 300 chars: {content[:300]}")
                
                # Try to extract JSON from markdown code blocks
                # Pattern 1: ```json ... ``` with flexible whitespace
                json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    logging.debug(f"Found markdown code block")
                    try:
                        parsed = json.loads(json_str)
                        logging.info(f"✓ Successfully parsed JSON from markdown code block with {len(parsed)} fields")
                        return parsed
                    except json.JSONDecodeError as e:
                        logging.warning(f"✗ Failed to parse JSON from markdown block: {e}")
                
                # Pattern 2: Try to find any JSON object/array in the content
                # Look for content that starts with { or [
                json_start = content.find('{')
                if json_start >= 0:
                    logging.debug(f"Found {{ at position {json_start}, attempting brace-matching extraction")
                    # Try to extract from the first { to a closing }
                    try:
                        # Find matching closing brace
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        
                        for i, char in enumerate(content[json_start:], start=json_start):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_str = content[json_start:i+1]
                                        parsed = json.loads(json_str)
                                        logging.info(f"✓ Successfully parsed JSON using brace-matching with {len(parsed)} fields")
                                        return parsed
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.warning(f"✗ Failed to extract JSON using brace-matching: {e}")
                
                # If no JSON found, return error dict
                logging.error(f"✗ Could not parse JSON from BaseMessage content (length: {len(content)})")
                logging.error(f"Full content:\n{content}")
                return {"error": "No JSON found in response", "raw_response": content[:1000]}
            
            # If it's already a Pydantic model instance, convert to dict
            if hasattr(response, 'model_dump'):
                logging.debug(f"Converting Pydantic model_dump()")
                return response.model_dump()
            elif hasattr(response, 'dict') and not isinstance(response, str):
                logging.debug(f"Converting Pydantic dict()")
                return response.dict()
            # If it's a string, try to parse as JSON
            elif isinstance(response, str):
                logging.debug(f"Parsing string as JSON")
                return json.loads(response)
            else:
                # Try to convert to dict if possible
                logging.warning(f"Response type {type(response)} not recognized")
                if hasattr(response, '__dict__'):
                    return response.__dict__
                return response
        except Exception as e:
            logging.error(f"✗ Error parsing response: {e}. Response type: {type(response)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "response_type": str(type(response))}
    
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
            logging.info(f"retrieve_documents called with query: {query[:50]}... and model_doc: {product_model_doc}")
            
            # First, try without filters to see if documents exist at all
            results = self.vector_db.query(query, top_k=5, filters=None)
            logging.info(f"Query without filter returned {len(results)} results")
            
            # If no results without filter, return empty
            if not results:
                logging.warning(f"No documents found in VectorDB for query: {query}")
                return "No policy documents found in database."
            
            # If results found, try to filter by product model if specified
            if product_model_doc != "unknown" and results:
                # Try different filter path formats
                filter_paths = [
                    product_model_doc,
                    os.path.basename(product_model_doc),
                ]
                
                filtered_results = []
                for filter_path in filter_paths:
                    try:
                        filters = {"source": {"$eq": filter_path}}
                        filtered_results = self.vector_db.query(query, top_k=5, filters=filters)
                        if filtered_results:
                            logging.info(f"Found {len(filtered_results)} documents with filter: {filter_path}")
                            results = filtered_results
                            break
                    except Exception as e:
                        logging.debug(f"Filter path {filter_path} failed: {e}")
                        continue
            
            docs = [doc.page_content for doc in results]
            result_text = "\n\n".join([f"Policy {i+1}: {doc}" for i, doc in enumerate(docs)])
            logging.info(f"Returning {len(docs)} policy documents ({len(result_text)} chars)")
            return result_text
        
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
            confidence_score=1.0,
            reviewer_id="AI-Agent",
            claim_id=f"SPAM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            review_timestamp=datetime.now().isoformat(),
            notes="Spam email - no processing needed"
        )
        
        return {"review_packet": packet}
    
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
            confidence_score=0.8 if can_answer else 0.6,
            reviewer_id="AI-Agent",
            claim_id=f"INQUIRY-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            review_timestamp=datetime.now().isoformat(),
            notes=f"Non-warranty inquiry. {'Answered from documentation.' if can_answer else 'Requires specialist input.'}"
        )
        
        return {
            "review_packet": packet,
            "customer_response": customer_response
        }
    
    def retrieve_policies(self, state: RAGState):
        """Retrieve relevant policy documents using tools"""
        
        claim_info = state["claim_info"]
        
        # Extract information from claim_info dictionary
        issue_description = claim_info.get("issue_description", claim_info.get("Issue Description", ""))
        product_model = claim_info.get("product_model", claim_info.get("Product Model", ""))
        
        # If product_model not found, try to extract from processed_output
        if not product_model or product_model == "unknown":
            processed_output = claim_info.get("processed_output", "")
            # Look for patterns like "Product Model: BLD-150" or "Model: BLD-150"
            import re
            # Pattern to match model codes like BLD-150, BLD-2000, etc.
            model_patterns = [
                r"(?:Product\s+)?Model[:\s]+([A-Z]+-\d+(?:\s+[A-Za-z0-9-]*)?)",  # Matches "BLD-150" or "Model: BLD-150 Everyday"
                r"Model[:\s]+([A-Z]+-[\d]+)",  # Matches "Model: BLD-150"
                r"(\b[A-Z]{3}-\d+\b)",  # Matches standalone BLD-150 pattern
            ]
            for pattern in model_patterns:
                try:
                    match = re.search(pattern, processed_output, re.IGNORECASE)
                    if match:
                        product_model = match.group(1).strip()
                        logging.info(f"Extracted product model from processed_output: {product_model}")
                        break
                except re.error as e:
                    logging.warning(f"Regex pattern error: {e}")
                    continue
        
        directory = config["POLICY_MANUALS"]["directory"]
        
        # Find file in directory matching product model
        product_doc = "unknown"
        if product_model and product_model != "unknown":
            # Extract just the model code (e.g., "BLD-150" from "BLD-150 Everyday")
            model_code = product_model.split()[0] if product_model else ""
            
            for file in os.listdir(directory):
                # Case-insensitive match for the product model in filename
                file_lower = file.lower()
                model_lower = model_code.lower()
                if model_lower and (model_lower in file_lower or file_lower.startswith(model_lower)):
                    product_doc = file
                    logging.info(f"Matched policy document: {product_doc} for model: {model_code}")
                    break
                    
        
        # If still not found, use a generic/default policy
        if product_doc == "unknown":
            # Try to find any BreezeLite policy as fallback
            for file in os.listdir(directory):
                if file.endswith(".pdf"):
                    product_doc = file
                    break
        
        product_doc_path = os.path.join(directory, product_doc) if product_doc != "unknown" else "unknown"
        logging.info(f"Product document path for retrieval: {product_doc_path}")
        
        # If claim_info contains processed_output, extract from there
        if not issue_description and "processed_output" in claim_info:
            issue_description = claim_info.get("processed_output", "")
        
        # Use tool to retrieve documents
        messages = [
            SystemMessage(content="You are a policy document retrieval assistant. Use the retrieve_documents tool to find relevant warranty policies, by passing the matched policy document path."),
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
            # Fallback: directly call tool with the full product_doc_path
            docs_text = self.tools[0].invoke({
                "query": issue_description,
                "product_model_doc": product_doc_path if product_doc_path != "unknown" else "unknown"
            })
            retrieved_docs.append(docs_text)
        
        return {"policy_doc_selected": product_doc_path, "retrieved_docs": retrieved_docs}
    
    def analyze_against_policy(self, state: RAGState):
        """Analyze claim against retrieved policy documents"""
        
        claim_info = state["claim_info"]
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Ensure retrieved_docs is a list
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [str(retrieved_docs)] if retrieved_docs else []
        
        logging.info(f"Analyzing claim with {len(retrieved_docs)} policy documents")
        
        # Create policy text from retrieved docs
        if retrieved_docs:
            policies_text = "\n".join(retrieved_docs)
        else:
            logging.warning("No retrieved documents for policy analysis")
            policies_text = "No specific policies available - will provide general warranty analysis based on claim information."
        
        analysis_prompt = """You are a warranty policy analyst. Analyze the claim against the warranty policies and respond with valid JSON.

Provide analysis as JSON with fields:
{
  "warranty_period_valid": true/false,
  "issue_covered": true/false,
  "evidence_sufficient": true/false,
  "exclusions_apply": true/false,
  "summary": "analysis summary",
  "key_findings": ["finding1", "finding2"]
}

Check:
1. Is the claim within the warranty period?
2. Is the reported issue covered by warranty?
3. Is the evidence sufficient (valid receipt, damage proof)?
4. Do any exclusions apply?"""

        context = f"""
Claim Information:
{json.dumps(claim_info, indent=2)}

Relevant Warranty Policies:
{policies_text}
"""

        logging.debug(f"Policy analysis context:\n{context[:200]}...")
        
        analysis = self.policy_analyzer.invoke([
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=context)
        ])
        
        # Parse response handling both Pydantic models and raw JSON
        parsed_analysis = self._parse_response(analysis)
        logging.info(f"Policy analysis result: {parsed_analysis}")
        return {"policy_analysis": parsed_analysis}
    
    def generate_recommendation(self, state: RAGState):
        """Generate structured recommendation based on analysis"""
        
        claim_info = state["claim_info"]
        policy_analysis = state.get("policy_analysis", {})
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Ensure retrieved_docs is a list
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [str(retrieved_docs)] if retrieved_docs else []
        
        logging.info(f"Generating recommendation with {len(retrieved_docs)} policy documents")
        
        # Create policy text from retrieved docs
        if retrieved_docs:
            policies_text = "\n".join(retrieved_docs)
        else:
            logging.warning("No retrieved documents available for recommendation generation")
            policies_text = "No specific warranty policies available - will provide general warranty guidance."
        
        recommendation_prompt = """You are a warranty claim decision specialist. Based on the claim information and policy analysis, provide a structured recommendation in valid JSON format.

Your response MUST be valid JSON with ALL the following fields:
{
  "claim_validity": "Valid|Invalid|Uncertain|N/A",
  "warranty_coverage": "Covered|Not Covered|Partially Covered|N/A",
  "decision": "Approve Claim|Reject Claim|Escalate for Further Review|Ignore|Provide Information",
  "confidence_score": 0.0-1.0,
  "reasons": ["reason1", "reason2", "reason3"],
  "next_steps": ["step1", "step2"],
  "notes": "summary text"
}

Decision Logic:
- claim_validity: Determine if the reported issue is a valid warranty claim
- warranty_coverage: "Covered" if within warranty period AND policy covers it; "Not Covered" if excluded or expired; "Partially Covered" if partial coverage
- decision: Choose based on validity and coverage - Approve if Covered, Reject if Not Covered
- confidence_score: 0.9-1.0 for clear decisions, 0.6-0.8 for borderline cases
- reasons: List 2-3+ key justifications for your decision
- next_steps: Provide 1-2+ actionable follow-ups
- notes: Concise summary of key facts and decision rationale

Consider: Policy compliance, evidence quality, safety implications, customer service impact."""

        context = f"""
Claim Information:
{json.dumps(claim_info, indent=2)}

Policy Analysis:
{json.dumps(policy_analysis, indent=2)}

Warranty Policies Retrieved:
{policies_text}
"""

        logging.debug(f"Recommendation context:\n{context[:200]}...")
        
        recommendation = self.recommender.invoke([
            SystemMessage(content=recommendation_prompt),
            HumanMessage(content=context)
        ])
        
        # Log raw response for debugging
        if hasattr(recommendation, 'content'):
            logging.info(f"Raw recommendation response (first 500 chars): {recommendation.content[:500]}")
            logging.debug(f"Full recommendation response (first 1000 chars): {recommendation.content[:1000]}")
        else:
            logging.info(f"Recommendation response type: {type(recommendation)}, content: {str(recommendation)[:500]}")
        
        # Parse response handling both Pydantic models and raw JSON
        parsed_recommendation = self._parse_response(recommendation)
        logging.info(f"Parsed recommendation keys: {list(parsed_recommendation.keys())}")
        logging.info(f"Parsed recommendation values: {parsed_recommendation}")
        logging.info(f"Generated recommendation: {parsed_recommendation.get('decision', 'Unknown')}")
        return {"recommendation": parsed_recommendation}
    
    def create_review_packet(self, state: RAGState):
        """Create structured human review packet"""
        
        recommendation = state["recommendation"]
        claim_info = state["claim_info"]
        
        logging.info(f"create_review_packet: recommendation type: {type(recommendation)}")
        logging.info(f"create_review_packet: recommendation keys: {list(recommendation.keys()) if isinstance(recommendation, dict) else 'N/A'}")
        
        # Check if recommendation has the required fields
        if "decision" not in recommendation and "claim_validity" not in recommendation:
            logging.error(f"Recommendation missing required fields. Content: {recommendation}")
            # It might be a raw response that wasn't parsed properly
            if "raw_response" in recommendation:
                logging.error(f"Raw response was: {recommendation['raw_response'][:500]}")
        
        # Ensure all fields have values with sensible defaults
        claim_validity = recommendation.get("claim_validity", "Uncertain")
        warranty_coverage = recommendation.get("warranty_coverage", "N/A")
        decision = recommendation.get("decision", "Escalate for Further Review")
        reasons = recommendation.get("reasons", ["Insufficient information for decision"])
        confidence_score = recommendation.get("confidence_score", 0.5)
        next_steps = recommendation.get("next_steps", [])
        notes = recommendation.get("notes", "")
        
        logging.info(f"create_review_packet: claim_validity={claim_validity}, decision={decision}, confidence={confidence_score}")
        
        # Validate confidence_score is numeric
        if not isinstance(confidence_score, (int, float)):
            confidence_score = 0.5
        
        # Ensure lists are actual lists
        if not isinstance(reasons, list):
            reasons = [str(reasons)] if reasons else ["Insufficient information"]
        
        if not isinstance(next_steps, list):
            next_steps = [str(next_steps)] if next_steps else ["Review claim with human agent"]
        
        # Provide defaults if empty
        if not next_steps:
            next_steps = ["Review claim with human agent", "Update customer on status"]
        
        if not notes:
            notes = f"Claim decision: {decision}. Validity: {claim_validity}. Coverage: {warranty_coverage}"
        
        # Create review packet
        packet = HumanReviewPacket(
            claim_validity=claim_validity,
            warranty_coverage=warranty_coverage,
            decision=decision,
            reasons=reasons,
            confidence_score=confidence_score,
            next_steps=next_steps,
            reviewer_id="AI-Agent",
            claim_id=f"CLAIM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            review_timestamp=datetime.now().isoformat(),
            notes=notes
        )
        
        logging.info(f"create_review_packet: Created packet with decision={packet.decision}")
        return {"review_packet": packet}
    
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
        
        # Handle review_packet - it's already a HumanReviewPacket object from the workflow
        packet = state["review_packet"]
        
        # If it's a dict (for backward compatibility), convert to HumanReviewPacket
        if isinstance(packet, dict):
            packet = HumanReviewPacket(**packet)
        
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