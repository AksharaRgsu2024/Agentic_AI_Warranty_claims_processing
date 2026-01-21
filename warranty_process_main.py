from claims_processing_agent import EmailTriageAgent
from RAG_recommender_agent import RAGRecommendationAgent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.graph import StateGraph, START, END
from typing import List, Optional
from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import json
import time
from Mailbox_mockup import MockMailboxAdapter
import re
import logging

class EmailState(TypedDict, total=False):
    subject: str
    body: str
    from_email: str
    date: str
    attachments: list
    category: str
    confidence: float
    reasoning: str
    processed_output: str
    attachment_analysis: str

# State for the RAG workflow
class RAGState(TypedDict, total=False):
    claim_info: dict
    policy_doc_selected: str
    retrieved_docs: list
    policy_analysis: str
    recommendation: dict
    review_packet: dict

class PipelineState(EmailState, RAGState):
    customer_email: dict
    triage_category: str
    is_warranty: bool
    email_info: dict
    claims_process_confidence: float
    recommendation: dict
    review_packet: dict
    reviewer_id: Optional[str] = None       # Human reviewer's ID or name
    review_timestamp: Optional[str] = None  # ISO8601 timestamp of review
    human_response: str
    human_feedback: str
    final_action:str
    response_email: str
    return_label: str

def claim_processing_agent(state: PipelineState) -> PipelineState:
    logging.info(" Claim email processing node running..... ")
    agent = EmailTriageAgent(
        attachments_folder="inbox"  # Folder where emails are saved
    )
    processed_info=agent.process_email(state["customer_email"])
    print("="*70 + "\n")
    print("Output of Claims Processing Agent:")
    for item, value in processed_info.items():
        print(f"{item}:\n {value}")
              
    category=processed_info["category"]
    state["triage_category"]=category
    if category=="warranty":
        state["is_warranty"]=True
        
    else:
        state["is_warranty"]=False
    state["claims_process_confidence"]=processed_info["confidence"]
    state["email_info"]=processed_info["processed_output"]
    return state

def route_decision(state: PipelineState) ->  Literal["display_spam", "rag_recommendation_agent", "display_non_warranty" ]
        """Conditional edge function to route based on category"""
        category = state["triage_category"]
        
        if category == "spam":
            return "display_spam"
        elif category == "warranty":
            return "rag_recommendation_agent"
        elif category == "non_warranty":
            return "display_non_warranty"

def rag_recommendation_agent(state: PipelineState) -> PipelineState:
    logging.info(" RAG recommendation node running..... ")
    rag_agent=RAGRecommendationAgent()
    review_packet = rag_agent.process_claim(state["email_info"])
    state["review_packet"]=review_packet

def display_spam(state: PipelineState)-> PipelineState:
    print("SPAM Email received - Ignoring")
    print(state["email_info"])

def display_non_warranty(state: PipelineState) -> PipelineState:
    print("Non-warranty wmail received, displaying for reference:")
    print(state["email_info"])

def get_human_feedback(state: PipelineState) -> PipelineState:
    """Collect feedback from a human reviewer"""
    # In a real application, this would be implemented with a UI
    # or messaging platform to collect human input
    print("\n👋 HUMAN REVIEW REQUIRED!\n")
    print(f"Processed information from claims email:\n {state['email_info']}")

    print(f"Recommendations of RAG Agent:\n {state['review_packet']}")
    
    # Simulating human input via console
    human_decision = input("\nPlease review and type 'approve' to accept, 'reject' to reject and 'revise' to re-run the workflow: ")
    feedback = input("Enter any specific feedback or comments if needed: ")
    print(f"\n👤 Human provided decision: {human_decision}\n")
    state["human_response"]= human_decision
    state["human_feedback"]= feedback

    return state

def final_action(state: PipelineState) ->Literal["draft_response", "rerun_workflow", "abort"]:
    if state["human_response"]=="approve":
        return "draft_response"
    elif state["human_response"]=="revise":
        return "rerun_workflow"
    else:
        return "abort"


def draft_response(state: PipelineState) ->PipelineState:
    claims_info=state["email_info"]
    recommendations=state["review_packet"]
    prompt="""
    You are a customer claims warranty service assistant. Draft a response email for the customer's claims , with the essential information given below:
    processed claims information from customer:
    
    Processed information from customer email:
    {claims_info}

    Recommendations:
    {recommendations}

    Instructions:
    
    """


def build_workflow():
    """Build the LangGraph workflow"""
    orchestrator=StateGraph(PipelineState)
    orchestrator.add_edge("START", "claim_processing_agent")
    orchestrator.add_node("claim_processing_agent", claim_processing_agent)
    orchestrator.add_conditional_edges("claim_processing_agent", route_decision, 
                                       {"display_spam": "display_spam",
                                         "rag_recommendation_agent":"rag_recommendation_agent",
                                         "display_non_warranty": "display_non_warranty"})
    
    orchestrator.add_edge("display_spam", "END")
    orchestrator.add_edge("display_non_warranty", "END")
    orchestrator.add_node("get_human_feedback", get_human_feedback)
    orchestrator.add_edge( "rag_recommendation_agent", "get_human_feedback")
    orchestrator.add_conditional_edges("get_human_feedback", final_action)