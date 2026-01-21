"""
Email Triage Agent with Router Pattern using LangGraph and Groq
Classifies emails and routes non-spam emails for further processing
Supports image attachment analysis
"""

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import os
import base64
from pathlib import Path
import json
import time
from Mailbox_mockup import MockMailboxAdapter
from dotenv import load_dotenv
load_dotenv()

# Schema for routing decision
class Route(BaseModel):
    step: Literal["spam", "warranty", "non_warranty"] = Field(
        description="The category of the email: spam, warranty, or non_warranty"
    )



# State for the workflow
class EmailState(TypedDict):
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


class EmailTriageAgent:
    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", attachments_folder: str = "processed", groq_api_key: str = None):
        """
        Initialize the email triage agent with router workflow
        
        Args:
            groq_api_key: Your Groq API key
            model: Groq model to use (vision model for image support)
            attachments_folder: Folder where attachments are saved
        """
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model
        )
        
        # Create router with structured output
        self.router = self.llm.with_structured_output(Route)
        
        self.attachments_folder = Path(attachments_folder)
        

        # Create tools
        self.tools = self._create_tools()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the workflow
        self.workflow = self._build_workflow()
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_tools(self):
        """Create tools for the agent"""
        
        @tool
        def analyze_image_attachment(filename: str, analysis_type: str = "general") -> str:
            """
            Analyze an image attachment from the email.
            
            Args:
                filename: Name of the image file to analyze
                analysis_type: Type of analysis - 'receipt', 'damage', or 'general'
            
            Returns:
                Detailed analysis of the image content
            """
            image_path = self.attachments_folder / filename
            
            if not image_path.exists():
                return f"Error: File '{filename}' not found in {self.attachments_folder}"
            
            try:
                # Encode image
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Determine content type
                content_type = "image/png" if filename.endswith('.png') else "image/jpeg"
                
                # Create analysis prompt based on type
                if analysis_type == "receipt":
                    prompt = f"Analyze this receipt image ({filename}). Extract: purchase date, product model, store name, price, and warranty period if visible."
                elif analysis_type == "damage":
                    prompt = f"Analyze this damage image ({filename}). Describe in detail: the type of damage visible, location of damage, severity, any safety hazards (sparks, burns, melting), and specific components affected."
                else:
                    prompt = f"Analyze this image ({filename}) and provide a detailed description of what you see."
                
                # Analyze image with vision model
                response = self.llm.invoke([
                    SystemMessage(content="You are analyzing images for warranty claims. Be thorough and specific in your descriptions."),
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}"
                            }
                        }
                    ])
                ])
                
                return response.content
                
            except Exception as e:
                return f"Error analyzing {filename}: {str(e)}"
        
        return [analyze_image_attachment]
    
    def analyze_attachments(self, attachments: list) -> str:
        """Analyze image attachments using vision model"""
        if not attachments:
            return "No attachments found."
        
        analysis_results = []
        
        for attachment in attachments:
            filename = attachment.get("file_name")
            content_type = attachment.get("content_type", "")
            
            # Only process images
            if not content_type.startswith("image/"):
                analysis_results.append(f"- {filename}: Skipped (not an image)")
                continue
            
            image_path = self.attachments_folder / filename
            
            if not image_path.exists():
                analysis_results.append(f"- {filename}: File not found in {self.attachments_folder}")
                continue
            
            try:
                # Encode image
                base64_image = self.encode_image(image_path)
                
                # Analyze image with vision model
                response = self.llm.invoke([
                    SystemMessage(content="You are analyzing images attached to customer warranty claims for a hairdryer company. Describe what you see in detail, focusing on any damage, defects, or issues visible in the image."),
                    HumanMessage(content=[
                        {
                            "type": "text",
                            "text": f"Analyze this image ({filename}):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}"
                            }
                        }
                    ])
                ])
                
                analysis_results.append(f"- {filename}:\n  {response.content}")
                
            except Exception as e:
                analysis_results.append(f"- {filename}: Error analyzing image - {str(e)}")
        
        return "\n".join(analysis_results)
    
    def classify_router(self, state: EmailState):
        """Router node: Classify the email into spam, warranty, or non-warranty"""
        
        # Include attachment info in classification
        attachment_info = ""
        if state.get("attachments"):
            attachment_names = [a.get("filename", "unknown") for a in state["attachments"]]
            attachment_info = f"\nAttachments: {', '.join(attachment_names)}"
        
        decision = self.router.invoke([
            SystemMessage(content="""You are an email classifier for a hairdryer company.
Classify emails into:
- spam: Promotional emails, phishing, unrelated content, scams
- warranty: Product defects, warranty claims, repair requests, malfunctions, safety issues
- non_warranty: General inquiries, product info, purchase questions, usage help, feedback

Emails with receipts or damage images are likely warranty claims.
Be strict with spam classification."""),
            HumanMessage(content=f"Subject: {state['subject']}\n\nBody: {state['body']}{attachment_info}")
        ])
        
        # Get confidence and reasoning
        response = self.llm.invoke([
            SystemMessage(content="Provide a confidence score (0-1) and brief reasoning for this classification."),
            HumanMessage(content=f"Email classified as: {decision.step}\nSubject: {state['subject']}\nBody: {state['body']}{attachment_info}")
        ])
        
        return {
            "category": decision.step,
            "confidence": 0.85,
            "reasoning": response.content
        }
    
    def process_warranty(self, state: EmailState):
        """Process warranty-related emails with tool-based image analysis"""
        
        # Build attachment list for the prompt
        attachment_list = []
        for attachment in state.get("attachments", []):
            filename = attachment.get("file_name", "unknown")
            attachment_list.append(filename)
        
        attachment_info = ", ".join(attachment_list) if attachment_list else "None"
        
        process_prompt = """You are a warranty support specialist for a hairdryer company.

Your task is to analyze warranty claims and validate them with evidence from attached images.

INSTRUCTIONS:
1. For each image attachment, use the analyze_image_attachment tool to examine it
   - For receipts: use analysis_type="receipt" 
   - For damage photos: use analysis_type="damage"
2. Extract all relevant information from the email and images
3. Validate the customer's claims against the visual evidence
4. Provide a comprehensive structured summary

REQUIRED OUTPUT FORMAT:
Customer Name: [extract from email body]
Contact Info: [email and phone if provided]
Customer address: [if available]
Product Model: [specific model mentioned]
Serial Number: [serial number of product]
Issue Description: [detailed description with severity level]
Purchase Date: [from receipt or email]
Evidence Provided: [list what images show]
Claim Validation: [Does visual evidence support the claims? Any discrepancies?]
Safety Concerns: [any hazards identified]
Priority Level: [Low/Medium/High/Urgent based on severity and safety]
Recommended Action: [next steps for support team]

Be thorough and reference specific details from the image analysis."""

        # Invoke LLM with tools
        messages = [
            SystemMessage(content=process_prompt),
            HumanMessage(content=f"""
Email Details:
Subject: {state['subject']}
From: {state.get('from_email', 'N/A')}
Date: {state.get('date', 'N/A')}

Body:
{state['body']}

Attachments to analyze: {attachment_info}

Please analyze all attachments using the analyze_image_attachment tool, then provide your comprehensive warranty claim summary.""")
        ]
        
        result = self.llm_with_tools.invoke(messages)
        
        # Check if tools were called and get their results
        tool_outputs = []
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tool_call in result.tool_calls:
                if tool_call['name'] == 'analyze_image_attachment':
                    tool_result = self.tools[0].invoke(tool_call['args'])
                    tool_outputs.append(f"\nTool Analysis ({tool_call['args']['filename']}):\n{tool_result}")
            
            # If tools were called, invoke again with tool results to get final summary
            if tool_outputs:
                messages.append(result)
                messages.append(HumanMessage(content="\n".join(tool_outputs) + "\n\nNow provide your final warranty claim summary based on the tool analysis."))
                result = self.llm.invoke(messages)
        
        # Format output
        output = f"WARRANTY CLAIM SUMMARY\n{'='*70}\n"
        output += f"From: {state.get('from_email', 'N/A')}\n"
        output += f"Date: {state.get('date', 'N/A')}\n\n"
        output += f"{result.content}\n"
        
        if tool_outputs:
            output += f"\n{'='*70}\nIMAGE ANALYSIS DETAILS\n{'='*70}"
            output += "\n".join(tool_outputs)
        
        return {"processed_output": output}
    
    def process_non_warranty(self, state: EmailState):
        """Process non-warranty customer inquiries"""
        
        result = self.llm.invoke([
            SystemMessage(content="""You are a customer service specialist for a hairdryer company.
Extract key information from customer inquiries:
- Customer details
- Type of inquiry (product info, purchase, usage, feedback, etc.)
- Specific questions asked
- Customer intent
- Suggested response approach

Format as a structured summary."""),
            HumanMessage(content=f"Subject: {state['subject']}\n\nFrom: {state.get('from_email', 'N/A')}\n\nBody: {state['body']}")
        ])
        
        output = f"CUSTOMER INQUIRY SUMMARY\n{'='*70}\n"
        output += f"From: {state.get('from_email', 'N/A')}\n"
        output += f"Date: {state.get('date', 'N/A')}\n"
        output += f"\n{result.content}"
        
        return {"processed_output": output}
    
    def handle_spam(self, state: EmailState):
        """Handle spam emails - minimal processing"""
        return {
            "processed_output": "SPAM - No action required. Email filtered."
        }
    
    def route_decision(self, state: EmailState):
        """Conditional edge function to route based on category"""
        category = state["category"]
        
        if category == "spam":
            return "handle_spam"
        elif category == "warranty":
            return "process_warranty"
        elif category == "non_warranty":
            return "process_non_warranty"
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        
        # Create graph
        builder = StateGraph(EmailState)
        
        # Add nodes
        builder.add_node("classify_router", self.classify_router)
        builder.add_node("handle_spam", self.handle_spam)
        builder.add_node("process_warranty", self.process_warranty)
        builder.add_node("process_non_warranty", self.process_non_warranty)
        
        # Add edges
        builder.add_edge(START, "classify_router")
        
        # Add conditional edges from router
        builder.add_conditional_edges(
            "classify_router",
            self.route_decision,
            {
                "handle_spam": "handle_spam",
                "process_warranty": "process_warranty",
                "process_non_warranty": "process_non_warranty"
            }
        )
        
        # All processing nodes lead to END
        builder.add_edge("handle_spam", END)
        builder.add_edge("process_warranty", END)
        builder.add_edge("process_non_warranty", END)
        
        # Compile and return
        return builder.compile()
    
    def process_email(self, email_data: dict) -> dict:
        """
        Process a single email through the workflow
        
        Args:
            email_data: Dictionary with email details (subject, body, from, date, attachments)
            
        Returns:
            Dictionary with classification and processed output
        """
        state = self.workflow.invoke({
            "subject": email_data.get("subject", ""),
            "body": email_data.get("body", ""),
            "from_email": email_data.get("from", ""),
            "date": email_data.get("date", ""),
            "attachments": email_data.get("attachments", [])
        })
        
        return {
            "category": state["category"],
            "confidence": state.get("confidence", 0.0),
            "reasoning": state.get("reasoning", ""),
            "processed_output": state.get("processed_output", ""),
            "attachment_analysis": state.get("attachment_analysis", "")
        }


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = EmailTriageAgent(
        attachments_folder="test_customer_emails"  # Folder where emails are saved
    )
    
    adapter = MockMailboxAdapter(
        inbox_dir="test_customer_emails",
        poll_interval=2.0,
    )

    
    print("EMAIL TRIAGE WITH VLM IMAGE PROCESSING")
    print("="*70 + "\n")
    for i, email in enumerate(adapter.read_all_once()):
        print("Received email subject:", email.get("subject"))
        result = agent.process_email(email)
  
    
        print(f"CATEGORY: {result['category'].upper()}")
        print(f"CONFIDENCE: {result['confidence']:.2f}")
        print(f"\nREASONING:\n{result['reasoning']}")
        print(f"\n{result['processed_output']}")
        print("\n" + "="*70)

    