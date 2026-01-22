"""
Warranty Claims Processing - Streamlit UI
Interactive web interface for processing warranty claims using AI agents
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import time
from dotenv import load_dotenv
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Warranty Claims AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Import agents
from claims_processing_agent import EmailTriageAgent
from RAG_recommender_agent import RAGRecommendationAgent
from Mailbox_mockup import MockMailboxAdapter

# Initialize session state
if "current_email" not in st.session_state:
    st.session_state.current_email = None
if "triage_result" not in st.session_state:
    st.session_state.triage_result = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None
if "processed_emails" not in st.session_state:
    st.session_state.processed_emails = []
if "final_drafted_response" not in st.session_state:
    st.session_state.final_drafted_response = None
if "review_submitted" not in st.session_state:
    st.session_state.review_submitted = False

# Load configuration
from configparser import ConfigParser
config = ConfigParser()
config.read('model_config.ini')

# Sidebar navigation
st.sidebar.title("🏢 Warranty Claims AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📧 Process Claim", "⏳ Review Queue", "📋 History", "⚙️ Settings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### About
    An intelligent multi-agent system for automated warranty claim processing.
    
    - **Email Triage**: Categorize incoming emails
    - **RAG Analysis**: Policy-based recommendations  
    - **Human Review**: Interactive decision making
    - **Response Generation**: Automated email drafting
    """
)


# Helper function to build summary from decision logs
def build_summary_from_decisions():
    """Reconstruct summary statistics from all decision log files"""
    summary = {
        "total_emails_processed": 0,
        "triage_breakdown": {
            "warranty": 0,
            "billing": 0,
            "technical": 0,
            "other": 0
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
    
    confidence_scores = []
    
    # Load all review files
    review_files = list(Path("decision_logs").glob("review_*.json"))
    
    for review_file in review_files:
        try:
            with open(review_file) as f:
                review = json.load(f)
            
            summary["total_emails_processed"] += 1
            
            # Extract triage category
            triage = review.get("triage_result", {})
            category = triage.get("category", "other").lower()
            if category in summary["triage_breakdown"]:
                summary["triage_breakdown"][category] += 1
            
            # Extract confidence
            triage_conf = triage.get("confidence", 0)
            if triage_conf > 1:
                triage_conf = triage_conf / 100
            if triage_conf > 0:
                confidence_scores.append(triage_conf)
                summary["confidence_stats"]["highest_confidence"] = max(
                    summary["confidence_stats"]["highest_confidence"], triage_conf
                )
                summary["confidence_stats"]["lowest_confidence"] = min(
                    summary["confidence_stats"]["lowest_confidence"], triage_conf
                )
            
            # Count decisions
            reviewer_decision = review.get("reviewer_decision", "").lower()
            if "approve" in reviewer_decision:
                summary["claim_decisions"]["approved"] += 1
            elif "reject" in reviewer_decision:
                summary["claim_decisions"]["rejected"] += 1
            elif "escalate" in reviewer_decision:
                summary["claim_decisions"]["escalated"] += 1
            else:
                summary["claim_decisions"]["not_applicable"] += 1
                
        except Exception as e:
            continue
    
    # Calculate average confidence
    if confidence_scores:
        summary["confidence_stats"]["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    return summary


def generate_response_email(recommendation: dict, reviewer_decision: str, reviewer_feedback: str, triage_result: dict, customer_response: str = None) -> dict:
    """Create a drafted response email based on AI recommendation and final reviewer decision."""
    # Basic templating - keeps things deterministic and editable
    to_addr = triage_result.get("From") or triage_result.get("from") or "customer@example.com"
    subject = f"Re: {triage_result.get('Subject', triage_result.get('subject', 'Warranty Claim'))}"

    # Build body
    lines = []
    lines.append(f"Hello,")
    lines.append("")
    # If AI provided a customer_response (for non-warranty), prefer that
    if customer_response:
        lines.append(customer_response)
    else:
        decision = reviewer_decision or recommendation.get("decision", "N/A")
        validity = recommendation.get("claim_validity", "N/A")
        coverage = recommendation.get("warranty_coverage", "N/A")

        lines.append(f"Thank you for contacting us about your product. Our review indicates: {validity} (Coverage: {coverage}).")
        lines.append("")
        if "approve" in decision.lower() or "approve" in recommendation.get("decision", "").lower():
            lines.append("We are pleased to inform you that your claim has been approved.")
            lines.append("Next Steps: " + ", ".join(recommendation.get("next_steps", ["Our team will contact you with a replacement or refund instructions."])) )
        elif "reject" in decision.lower():
            lines.append("We regret to inform you that your claim is not covered under the warranty based on the provided information.")
            lines.append("Reason(s): " + ", ".join(recommendation.get("reasons", ["See policy details."])))
            if recommendation.get("next_steps"):
                lines.append("Next Steps: " + ", ".join(recommendation.get("next_steps")))
        elif "escalate" in decision.lower():
            lines.append("Your case has been escalated for further review by a specialist. We will update you within 48 hours.")
        else:
            lines.append("We have recorded your request and will follow up shortly with more information.")

        if reviewer_feedback:
            lines.append("")
            lines.append("Reviewer notes:")
            lines.append(reviewer_feedback)

    lines.append("")
    lines.append("Regards,")
    lines.append("Warranty Claims Team")

    body = "\n".join(lines)

    return {
        "to": to_addr,
        "subject": subject,
        "body": body,
        "generated_at": datetime.now().isoformat()
    }


# Page: Dashboard
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    # Build summary from decision logs
    summary = build_summary_from_decisions()
    
    with col1:
        st.metric("Total Processed", summary.get("total_emails_processed", 0))
    
    with col2:
        approved = summary.get("claim_decisions", {}).get("approved", 0)
        st.metric("Approved Claims", approved)
    
    with col3:
        avg_conf = summary.get("confidence_stats", {}).get("average_confidence", 0)
        # Handle confidence as either decimal (0-1) or percentage (0-100)
        if avg_conf > 1:
            avg_conf = avg_conf / 100
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    st.markdown("---")
    
    # Triage breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Triage Breakdown")
        triage_data = summary.get("triage_breakdown", {})
        if triage_data:
            df_triage = pd.DataFrame(list(triage_data.items()), columns=["Category", "Count"])
            st.bar_chart(df_triage.set_index("Category"))
    
    with col2:
        st.subheader("Claim Decisions")
        decision_data = summary.get("claim_decisions", {})
        if decision_data:
            df_decisions = pd.DataFrame(list(decision_data.items()), columns=["Decision", "Count"])
            fig = px.pie(df_decisions, values="Count", names="Decision")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent decisions
    st.subheader("Recent Decisions")
    claim_files = sorted(Path("decision_logs").glob("claim_decisions_*.json"), reverse=True)
    if claim_files:
        with open(claim_files[0]) as f:
            recent_claims = json.load(f)
        
        if recent_claims:
            df_recent = pd.DataFrame(recent_claims[:5])
            st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No decisions recorded yet")


# Page: Process Claim
elif page == "📧 Process Claim":
    st.title("📧 Process Warranty Claim")
    
    # Display generated email at the top if it exists
    if st.session_state.get("review_submitted", False) and st.session_state.get("final_drafted_response"):
        st.success("✅ Review submitted successfully!")
        st.markdown("---")
        st.subheader("✉️ Generated Response Email")
        drafted = st.session_state.final_drafted_response
        st.write(f"**To:** {drafted.get('to', 'N/A')}")
        st.write(f"**Subject:** {drafted.get('subject', 'N/A')}")
        st.text_area("Message Body:", value=drafted.get('body', ''), height=320, disabled=True, key="top_email_display")
        st.info("✅ Workflow Complete! Email draft is ready to send.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Process Another Claim"):
                st.session_state.review_submitted = False
                st.session_state.final_drafted_response = None
                st.session_state.current_email = None
                st.session_state.triage_result = None
                st.session_state.recommendation = None
                st.session_state.customer_response = None
                st.rerun()
        with col_b:
            if st.button("📧 View in Review Queue"):
                st.session_state.review_submitted = False
                st.session_state.final_drafted_response = None
        
        st.markdown("---")
        st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Email input options
        input_method = st.radio("How would you like to input the claim?", 
                               ["Paste Email Text", "Load Test Email", "Upload JSON"])
        
        email_data = None
        
        if input_method == "Paste Email Text":
            email_text = st.text_area("Paste email content:", height=200)
            if email_text and st.button("Process Email"):
                email_data = {
                    "Subject": "Customer Claim",
                    "From": "customer@example.com",
                    "Body": email_text,
                    "Date": datetime.now().isoformat(),
                    "attachment_analysis": "N/A"
                }
        
        elif input_method == "Load Test Email":
            test_emails = sorted(Path("test_customer_emails").glob("email_*.json"))
            if test_emails:
                selected_email = st.selectbox(
                    "Select a test email:",
                    test_emails,
                    format_func=lambda x: x.name
                )
                
                # Display selected email preview
                try:
                    with open(selected_email) as f:
                        email_preview = json.load(f)
                    st.markdown("---")
                    st.subheader("📧 Email Preview")
                    st.write(f"**From:** {email_preview.get('from', 'N/A')}")
                    st.write(f"**Subject:** {email_preview.get('subject', 'N/A')}")
                    st.write(f"**Date:** {email_preview.get('date', 'N/A')}")
                    with st.expander("📝 Email Body"):
                        st.text(email_preview.get('body', 'No body content'))
                except Exception as e:
                    st.error(f"Error loading email preview: {e}")
                
                if st.button("Load & Process"):
                    with open(selected_email) as f:
                        email_data = json.load(f)
                    email_data["filename"] = selected_email.name
            else:
                st.warning("No test emails found in test_customer_emails/")
        
        elif input_method == "Upload JSON":
            uploaded_file = st.file_uploader("Upload email JSON file:", type="json")
            if uploaded_file and st.button("Process Email"):
                email_data = json.load(uploaded_file)
    
    with col2:
        st.markdown("### Processing Options")
        groq_key_set = bool(os.getenv("GROQ_API_KEY"))
        st.info(f"Groq API: {'✅ Configured' if groq_key_set else '❌ Not configured'}")
    
    # Process email if data is provided (but not if we just submitted a review)
    if email_data and not st.session_state.review_submitted:
        st.session_state.current_email = email_data
        
        with st.spinner("🔄 Running email triage..."):
            try:
                groq_api_key = os.getenv("GROQ_API_KEY")
                vlm_model = config["MODEL_CONFIG"]["vlm_model"]
                input_folder = config["EMAILS"]["customer_mails"]
                
                triage_agent = EmailTriageAgent(
                    groq_api_key=groq_api_key,
                    attachments_folder=input_folder,
                    model=vlm_model
                )
                
                triage_result = triage_agent.process_email(email_data)
                st.session_state.triage_result = triage_result
                st.success("✅ Triage complete!")

                # Automatically run RAG recommendation for warranty claims
                try:
                    category = triage_result.get("category", "").lower()
                    if category == "warranty":
                        with st.spinner("🔍 Running RAG recommendation..."):
                            groq_api_key = os.getenv("GROQ_API_KEY")
                            rag_model = config["MODEL_CONFIG"].get("rag_model", "")
                            rag_agent = RAGRecommendationAgent(
                                groq_api_key=groq_api_key,
                                model=rag_model
                            )
                            result = rag_agent.process_claim(st.session_state.triage_result)
                            review_packet = result.get("review_packet")
                            if hasattr(review_packet, 'to_dict'):
                                st.session_state.recommendation = review_packet.to_dict()
                            elif isinstance(review_packet, dict):
                                st.session_state.recommendation = review_packet
                            else:
                                # Fallback: try to convert
                                try:
                                    st.session_state.recommendation = review_packet.__dict__
                                except Exception:
                                    st.session_state.recommendation = {"notes": "Could not parse review packet"}
                            # Attach optional customer_response (for non-warranty answers)
                            if result.get("customer_response"):
                                st.session_state.customer_response = result.get("customer_response")
                            st.success("✅ Recommendation generated!")
                    else:
                        # For non-warranty, still run RAG to generate customer response if possible
                        with st.spinner("🔍 Running response generator..."):
                            groq_api_key = os.getenv("GROQ_API_KEY")
                            rag_model = config["MODEL_CONFIG"].get("rag_model", "")
                            rag_agent = RAGRecommendationAgent(
                                groq_api_key=groq_api_key,
                                model=rag_model
                            )
                            result = rag_agent.process_claim(st.session_state.triage_result)
                            if result.get("customer_response"):
                                st.session_state.customer_response = result.get("customer_response")
                except Exception as e:
                    st.error(f"❌ Error generating recommendation: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Error during triage: {str(e)}")
        
        # Display triage results (only if not just submitted)
        if st.session_state.triage_result and not st.session_state.review_submitted:
            st.markdown("---")
            st.subheader("📋 Triage Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.session_state.triage_result.get("category", "unknown").upper()
                st.metric("Category", category)
            with col2:
                confidence = st.session_state.triage_result.get("confidence", 0)
                # Handle confidence as either decimal (0-1) or percentage (0-100)
                if confidence > 1:
                    confidence = confidence / 100
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                product = st.session_state.triage_result.get("product_model", "Unknown")
                st.metric("Product Model", product)
            
            # Display full triage output
            with st.expander("📝 Full Triage Details"):
                for key, value in st.session_state.triage_result.items():
                    if key not in ["confidence"]:
                        if isinstance(value, str) and len(value) > 500:
                            st.text_area(f"{key}:", value, height=150, disabled=True)
                        else:
                            st.write(f"**{key}:** {value}")
            
            # If warranty claim, proceed to RAG
            if st.session_state.triage_result.get("category") == "warranty":
                st.markdown("---")
                
                # Display recommendation
                if st.session_state.recommendation:
                    st.subheader("✅ AI Recommendation")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Claim Validity", st.session_state.recommendation.get("claim_validity", "N/A"))
                    with col2:
                        st.metric("Warranty Coverage", st.session_state.recommendation.get("warranty_coverage", "N/A"))
                    with col3:
                        st.metric("Decision", st.session_state.recommendation.get("decision", "N/A"))
                    with col4:
                        conf = st.session_state.recommendation.get("confidence_score", 0)
                        # Handle confidence as either decimal (0-1) or percentage (0-100)
                        if conf > 1:
                            conf = conf / 100
                        st.metric("Confidence", f"{conf:.1%}")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📌 Reasons")
                        for i, reason in enumerate(st.session_state.recommendation.get("reasons", []), 1):
                            st.write(f"{i}. {reason}")
                    
                    with col2:
                        st.subheader("📋 Next Steps")
                        for i, step in enumerate(st.session_state.recommendation.get("next_steps", []), 1):
                            st.write(f"{i}. {step}")
                    
                    st.markdown("---")
                    
                    # Human review section
                    st.subheader("👤 Human Review")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        reviewer_decision = st.radio(
                            "Your Decision:",
                            ["Approve", "Reject", "Escalate", "Revise"],
                            horizontal=True
                        )
                    
                    with col2:
                        reviewer_feedback = st.text_area("Additional feedback (optional):")
                    
                    reviewer_name = st.text_input("Your Name/ID:", value="Anonymous Reviewer")
                    
                    if st.button("✅ Submit Review"):
                        with st.spinner("✉️ Generating response email..."):
                            try:
                                customer_resp = st.session_state.get("customer_response", None)
                                drafted = generate_response_email(
                                    recommendation=st.session_state.recommendation or {},
                                    reviewer_decision=reviewer_decision,
                                    reviewer_feedback=reviewer_feedback,
                                    triage_result=st.session_state.triage_result or {},
                                    customer_response=customer_resp
                                )
                                
                                # Record decision
                                decision_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "email_subject": email_data.get("Subject", "N/A"),
                                    "email_from": email_data.get("From", "N/A"),
                                    "reviewer_decision": reviewer_decision,
                                    "reviewer_feedback": reviewer_feedback,
                                    "reviewer_name": reviewer_name,
                                    "ai_recommendation": st.session_state.recommendation,
                                    "triage_result": st.session_state.triage_result,
                                    "final_response": drafted
                                }

                                # Save to files
                                os.makedirs("decision_logs", exist_ok=True)
                                os.makedirs("response_emails", exist_ok=True)
                                review_path = f"decision_logs/review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                resp_path = f"response_emails/response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                with open(review_path, "w") as f:
                                    json.dump(decision_record, f, indent=2)
                                with open(resp_path, "w") as f:
                                    json.dump(drafted, f, indent=2)

                                # Store in session state
                                st.session_state.final_drafted_response = drafted
                                st.session_state.review_submitted = True
                                
                                # Clear the processing state
                                st.session_state.current_email = None
                                
                                st.success(f"✅ Review recorded! Saved to {review_path}")
                                st.balloons()
                                
                                # Small delay to ensure state is saved
                                time.sleep(0.5)
                                
                                # Trigger rerun to show email at top
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error generating response: {str(e)}")


# Page: Review Queue
elif page == "⏳ Review Queue":
    st.title("⏳ Review Queue")
    
    st.info("Pending claims awaiting human review")
    
    # Load pending reviews
    review_files = sorted(Path("decision_logs").glob("review_*.json"), reverse=True)
    
    if review_files:
        # Create tabs for each pending review
        claims_data = []
        for review_file in review_files[:10]:  # Show last 10
            with open(review_file) as f:
                review_data = json.load(f)
                claims_data.append({
                    "Timestamp": review_data.get("timestamp", "N/A"),
                    "From": review_data.get("email_from", "N/A"),
                    "Subject": review_data.get("email_subject", "N/A"),
                    "AI Decision": review_data.get("ai_recommendation", {}).get("decision", "N/A"),
                    "Your Decision": review_data.get("reviewer_decision", "N/A"),
                    "Reviewer": review_data.get("reviewer_name", "N/A"),
                    "File": review_file.name
                })
        
        df_reviews = pd.DataFrame(claims_data)
        st.dataframe(df_reviews, use_container_width=True)
        
        # Select review to view details
        if st.checkbox("View detailed review"):
            selected_review = st.selectbox(
                "Select a review:",
                review_files,
                format_func=lambda x: x.name
            )
            
            with open(selected_review) as f:
                review_detail = json.load(f)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Email Info")
                st.write(f"From: {review_detail.get('email_from')}")
                st.write(f"Subject: {review_detail.get('email_subject')}")
                st.write(f"Timestamp: {review_detail.get('timestamp')}")
            
            with col2:
                st.subheader("Review Info")
                st.write(f"AI Decision: {review_detail.get('ai_recommendation', {}).get('decision')}")
                st.write(f"Your Decision: {review_detail.get('reviewer_decision')}")
                st.write(f"Reviewer: {review_detail.get('reviewer_name')}")
            
            if review_detail.get('reviewer_feedback'):
                st.subheader("Feedback")
                st.write(review_detail.get('reviewer_feedback'))
    else:
        st.info("No reviews recorded yet")


# Page: History
elif page == "📋 History":
    st.title("📋 Decision History")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        history_type = st.radio(
            "View:",
            ["Claim Decisions", "Triage Decisions", "All Reviews"],
            horizontal=True
        )
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Load appropriate files
    if history_type == "Claim Decisions":
        files = sorted(Path("decision_logs").glob("claim_decisions_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                data = json.load(f)
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("No claim decisions recorded yet")
    
    elif history_type == "Triage Decisions":
        files = sorted(Path("decision_logs").glob("triage_decisions_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                data = json.load(f)
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("No triage decisions recorded yet")
    
    elif history_type == "All Reviews":
        review_files = sorted(Path("decision_logs").glob("review_*.json"), reverse=True)
        if review_files:
            all_reviews = []
            for review_file in review_files:
                with open(review_file) as f:
                    review = json.load(f)
                    all_reviews.append({
                        "Timestamp": review.get("timestamp"),
                        "From": review.get("email_from"),
                        "Subject": review.get("email_subject"),
                        "AI Decision": review.get("ai_recommendation", {}).get("decision"),
                        "Your Decision": review.get("reviewer_decision"),
                        "Reviewer": review.get("reviewer_name")
                    })
            
            st.dataframe(pd.DataFrame(all_reviews), use_container_width=True)
        else:
            st.info("No reviews recorded yet")
    
    # Export option
    st.markdown("---")
    if st.button("📥 Export History to CSV"):
        files = sorted(Path("decision_logs").glob("*.json"), reverse=True)
        if files:
            all_data = []
            for file in files:
                try:
                    with open(file) as f:
                        all_data.append(json.load(f))
                except:
                    pass
            
            if all_data:
                df_export = pd.DataFrame(all_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"warranty_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )


# Page: Settings
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Model Configuration")
    
    try:
        with open("model_config.ini") as f:
            config_content = f.read()
        
        st.text_area("model_config.ini:", value=config_content, height=300, disabled=True)
    except:
        st.warning("Could not load model_config.ini")
    
    st.markdown("---")
    
    st.subheader("📊 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        decision_logs = len(list(Path("decision_logs").glob("*.json")))
        st.metric("Decision Logs", decision_logs)
    
    with col2:
        response_emails = len(list(Path("response_emails").glob("*.json")))
        st.metric("Response Emails", response_emails)
    
    with col3:
        test_emails = len(list(Path("test_customer_emails").glob("*.json")))
        st.metric("Test Emails", test_emails)
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    ### Warranty Claims AI Processing System
    
    **Version:** 1.0.0  
    **Framework:** Streamlit + LangGraph  
    **LLM Provider:** Groq (mixtral-8x7b-32768)  
    **Vector DB:** Pinecone  
    
    This application processes warranty claim emails through an intelligent multi-agent pipeline:
    
    1. **Email Triage** - Categorizes incoming emails
    2. **RAG Analysis** - Retrieves and analyzes policy documents
    3. **Recommendations** - Generates structured decisions with confidence scores
    4. **Human Review** - Interactive decision approval interface
    5. **Response Generation** - Drafts professional emails
    6. **Audit Trail** - Records all decisions for compliance
    
    For more information, see the README.md in the project directory.
    """)


if __name__ == "__main__":
    pass
