import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
from pathlib import Path

# Import our agents
from agents.coordinator_agent import CoordinatorAgent
from config.settings import Config
from utils.helpers import LoggingUtils

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic Recruitment System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = None
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def display_header():
    """Display the main header"""
    st.markdown('<div class="main-header">🤖 Agentic Recruitment System</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    Welcome to the AI-powered recruitment system that automatically matches candidates with job descriptions 
    using advanced vector similarity and multi-agent architecture.
    """)

def upload_files():
    """Handle file uploads"""
    st.header("📁 Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Description (PDF)")
        uploaded_jd = st.file_uploader(
            "Upload Job Description PDF",
            type=['pdf'],
            help="Upload a PDF file containing the job description"
        )
        
        if uploaded_jd:
            # Save uploaded file
            jd_path = Path(Config.UPLOAD_DIR) / f"jd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(jd_path, "wb") as f:
                f.write(uploaded_jd.read())
            st.success(f"✅ Job description uploaded: {uploaded_jd.name}")
            st.session_state.jd_path = str(jd_path)
    
    with col2:
        st.subheader("Candidates Data (CSV)")
        uploaded_csv = st.file_uploader(
            "Upload Candidates CSV",
            type=['csv'],
            help="Upload a CSV file with candidate information"
        )
        
        if uploaded_csv:
            # Save uploaded file
            csv_path = Path(Config.UPLOAD_DIR) / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_path, "wb") as f:
                f.write(uploaded_csv.read())
            st.success(f"✅ Candidates data uploaded: {uploaded_csv.name}")
            st.session_state.csv_path = str(csv_path)

            # Preview CSV from the saved file
            try:
                df = pd.read_csv(csv_path)
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())
                st.info(f"Total candidates: {len(df)}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    return uploaded_jd, uploaded_csv

def configure_processing():
    """Configure processing parameters"""
    st.header("⚙️ Processing Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for candidate matching"
        )
    
    with col2:
        max_matches = st.number_input(
            "Maximum Matches",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of top matches to return"
        )
    
    with col3:
        send_emails = st.checkbox(
            "Send Emails",
            value=False,
            help="Automatically send emails to matched candidates"
        )
    
    return similarity_threshold, max_matches, send_emails

def process_workflow():
    """Process the recruitment workflow"""
    st.header("🚀 Process Recruitment Workflow")
    
    # Check if files are uploaded
    if not hasattr(st.session_state, 'jd_path') or not hasattr(st.session_state, 'csv_path'):
        st.warning("⚠️ Please upload both job description (PDF) and candidates data (CSV) first.")
        return
    
    # Configuration
    similarity_threshold, max_matches, send_emails = configure_processing()
    
    # Process button
    if st.button("🔄 Start Processing", type="primary"):
        try:
            # Initialize coordinator
            config = Config()
            config.SIMILARITY_THRESHOLD = similarity_threshold
            config.TOP_K_MATCHES = max_matches
            
            coordinator = CoordinatorAgent(config)
            st.session_state.coordinator = coordinator
            
            # Prepare input data
            input_data = {
                'job_description_path': st.session_state.jd_path,
                'candidates_csv_path': st.session_state.csv_path,
                'send_emails': send_emails
            }
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Execute workflow
            status_text.text("🤖 Initializing agents...")
            progress_bar.progress(10)
            
            status_text.text("📄 Extracting job description and candidate data...")
            progress_bar.progress(30)
            
            status_text.text("🔍 Matching candidates with job requirements...")
            progress_bar.progress(60)
            
            if send_emails:
                status_text.text("📧 Sending emails to matched candidates...")
                progress_bar.progress(80)
            
            status_text.text("📊 Generating results and insights...")
            progress_bar.progress(90)
            
            # Process the workflow
            results = coordinator.process(input_data)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Store results
            st.session_state.workflow_results = results
            st.session_state.processing_complete = True
            
            # Display success message
            if results['success']:
                st.success("🎉 Workflow completed successfully!")
            else:
                st.error("❌ Workflow failed. Check the logs for details.")
                
        except Exception as e:
            st.error(f"❌ Error processing workflow: {str(e)}")
            st.exception(e)

def display_results():
    """Display workflow results"""
    if not st.session_state.processing_complete or not st.session_state.workflow_results:
        return
    
    results = st.session_state.workflow_results
    
    if not results['success']:
        st.error("❌ Workflow execution failed.")
        return
    
    st.header("📊 Results Dashboard")
    
    # Executive Summary
    workflow_summary = results['results']['workflow_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Candidates",
            workflow_summary.get('total_candidates_processed', 0)
        )
    
    with col2:
        st.metric(
            "Qualified Matches",
            workflow_summary.get('matches_found', 0)
        )
    
    with col3:
        st.metric(
            "Processing Time",
            f"{workflow_summary.get('execution_time_seconds', 0):.2f}s"
        )
    
    with col4:
        st.metric(
            "Emails Sent",
            workflow_summary.get('emails_sent', 0)
        )
    
    # Detailed Results
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Matches", "🔍 Analysis", "📧 Emails", "🤖 Agent Status"])
    
    with tab1:
        display_matches(results['results']['matching_results'])
    
    with tab2:
        display_analysis(results['results']['matching_results'])
    
    with tab3:
        display_email_results(results['results'].get('email_results'))
    
    with tab4:
        display_agent_status(results['results']['agent_status'])

def display_matches(matching_results):
    """Display candidate matches"""
    st.subheader("🎯 Top Candidate Matches")
    
    matches = matching_results.get('matches', [])
    
    if not matches:
        st.info("No matches found above the similarity threshold.")
        return
    
    # Create matches dataframe
    match_data = []
    for match in matches:
        candidate = match['candidate']
        match_data.append({
            'Rank': match['rank'],
            'Name': candidate['name'],
            'Email': candidate['email'],
            'Similarity Score': f"{match['similarity_score']:.2%}",
            'Skills': ', '.join(candidate.get('extracted_skills', [])[:3]),
            'Experience': candidate.get('experience', 'N/A')
        })
    
    df_matches = pd.DataFrame(match_data)
    st.dataframe(df_matches, use_container_width=True)
    
    # Similarity score distribution
    st.subheader("📊 Similarity Score Distribution")
    scores = [match['similarity_score'] for match in matches]
    
    fig = px.histogram(
        x=scores,
        nbins=10,
        title="Distribution of Similarity Scores",
        labels={'x': 'Similarity Score', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_analysis(matching_results):
    """Display matching analysis"""
    st.subheader("🔍 Matching Analysis")
    
    match_analysis = matching_results.get('match_analysis', {})
    
    if not match_analysis:
        st.info("No analysis data available.")
        return
    
    # Score statistics
    stats = match_analysis.get('score_statistics', {})
    if stats:
        st.subheader("📈 Score Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Score", f"{stats.get('max_score', 0):.2%}")
        with col2:
            st.metric("Min Score", f"{stats.get('min_score', 0):.2%}")
        with col3:
            st.metric("Avg Score", f"{stats.get('avg_score', 0):.2%}")
        with col4:
            st.metric("Std Dev", f"{stats.get('std_score', 0):.2%}")
    
    # Skill coverage
    skill_coverage = match_analysis.get('skill_coverage', {})
    if skill_coverage:
        st.subheader("💡 Skill Analysis")
        top_skills = skill_coverage.get('top_skills', [])[:10]
        
        if top_skills:
            skill_counts = []
            for skill in top_skills:
                coverage = skill_coverage.get('skill_coverage', {}).get(skill, 0)
                skill_counts.append({'Skill': skill, 'Coverage %': coverage})
            
            df_skills = pd.DataFrame(skill_counts)
            fig = px.bar(
                df_skills,
                x='Skill',
                y='Coverage %',
                title="Top Skills Coverage in Matches"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Insights and recommendations
    insights = match_analysis.get('insights', [])
    recommendations = match_analysis.get('recommendations', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Insights")
        for insight in insights:
            st.info(insight)
    
    with col2:
        st.subheader("📋 Recommendations")
        for rec in recommendations:
            st.warning(rec)

def display_email_results(email_results):
    """Display email sending results"""
    st.subheader("📧 Email Results")
    
    if not email_results:
        st.info("No emails were sent in this workflow.")
        return
    
    email_data = email_results.get('email_results', [])
    
    if not email_data:
        st.info("No email data available.")
        return
    
    # Email summary
    total_emails = len(email_data)
    sent_emails = sum(1 for email in email_data if email['status'] == 'sent')
    failed_emails = total_emails - sent_emails
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Emails", total_emails)
    with col2:
        st.metric("Sent Successfully", sent_emails)
    with col3:
        st.metric("Failed", failed_emails)
    
    # Email details
    st.subheader("📋 Email Details")
    df_emails = pd.DataFrame(email_data)
    st.dataframe(df_emails, use_container_width=True)

def display_agent_status(agent_status):
    """Display agent status information"""
    st.subheader("🤖 Agent Status")
    
    agents_status = agent_status.get('agents_status', {})
    
    for agent_id, status in agents_status.items():
        with st.expander(f"Agent: {status['name']}"):
            st.write(f"**Type:** {status['type']}")
            st.write(f"**Status:** {status['status']}")
            st.write(f"**Description:** {status['description']}")
            st.write(f"**Capabilities:** {', '.join(status['capabilities'])}")
            
            if status.get('last_action'):
                st.write("**Last Action:**")
                st.json(status['last_action'])

def display_comprehensive_report():
    """Display comprehensive report"""
    if not st.session_state.processing_complete or not st.session_state.coordinator:
        return

    st.header("📋 Comprehensive Report")

    try:
        print("DEBUG: Entered display_comprehensive_report")
        report = st.session_state.coordinator.get_comprehensive_report()
        print("DEBUG: Recommendations:", report.get('recommendations', []))
        print("DEBUG: Matches:", report.get('matching_results', {}).get('matches', []))
        print("DEBUG: get_comprehensive_report output:", report)
        # Executive Summary
        st.subheader("📊 Executive Summary")
        exec_summary = report.get('executive_summary', {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Candidates Evaluated", exec_summary.get('total_candidates_evaluated', 0))
            st.metric("Processing Time", exec_summary.get('processing_time', 'N/A'))
        with col2:
            st.metric("Qualified Matches Found", exec_summary.get('qualified_matches_found', 0))
            st.metric("Success Rate", f"{exec_summary.get('success_rate', 0):.1f}%")
        # Recommendations
        st.subheader("💡 System Recommendations")
        recommendations = report.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.info("No recommendations available for this workflow.")
        # Export matches as CSV only
        matches = report.get('matching_results', {}).get('matches', [])
        if not matches:
            st.info("No matches found for export.")
        else:
            match_data = []
            for match in matches:
                candidate = match['candidate']
                match_data.append({
                    'Rank': match.get('rank'),
                    'Name': candidate.get('name'),
                    'Email': candidate.get('email'),
                    'Similarity Score': f"{match.get('similarity_score', 0):.2%}",
                    'Skills': ', '.join(candidate.get('extracted_skills', [])[:3]),
                    'Experience': candidate.get('experience', 'N/A')
                })
            df_matches = pd.DataFrame(match_data)
            csv_data = df_matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Matches CSV",
                data=csv_data,
                file_name=f"recruitment_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

def main():
    """Main application function"""
    initialize_session_state()
    display_header()
    
    # Main workflow
    uploaded_jd, uploaded_csv = upload_files()
    
    if uploaded_jd and uploaded_csv:
        process_workflow()
    
    # Display results if available
    display_results()
    
    # Comprehensive report
    display_comprehensive_report()
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit, Sentence Transformers, and Multi-Agent Architecture**")

if __name__ == "__main__":
    main()
