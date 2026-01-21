from claims_processing_agent import EmailTriageAgent
from RAG_recommender_agent import RAGRecommendationAgent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
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



@tool
def process_claim_email(email: json):
    """
    Invokes the EmailTriageAgent to process the emails to traigae for warranty claims and extract information
    
    :param email: Customer email
    :type email: json
    """
    email_process_agent = EmailTriageAgent(
        attachments_folder="inbox"  # Folder where emails are saved
    )
    result = email_process_agent.process_email(email)

    return result

@tool 
def rag_recommendation(claims_info: str):
    """
    Invokes the RAGRecommendationAgent if the claims info indicates it is a warranty email. Spam and non-warranty emails are displayed to the user.
    
    :param claims_info: Claims information obtained from process_slaim_email tool
    :type claims_info: str
    """
    rag_agent = RAGRecommendationAgent()
    review_packet = rag_agent.process_claim(claims_info)
    return review_packet

SUPERVISOR_PROMPT = """ You are a Warranty Claims processing specialist for a hairdryer company.
 Your role is to assist the company with reviewing warranty claims, and reduce human workload by providing your recommendation for a human to review.
 
 Instructions:
    1. For the given customer email, use the process_claim_email tool to process the email and extract information.
    2. Check the claims information provided by the tool. If the category is"warranty", call the rag_recommendation tool using the claims information.
    If the category is "spam", display "SPAM Email received - Ignoring" and exit the workflow
    If the category is "Non-warranty", display "Non-warranty email received." . Display the processsed information and exit the workflow.

    3. Display the output of the rag_recommendation tool to the user to review. 
 
 """

orchestrator_agent=create_agent(middleware=[ 
        HumanInTheLoopMiddleware( 
            interrupt_on={"send_email": True}, 
            description_prefix="Outbound email pending approval", 
        ), 
    ], )