import os
import json
from io import BytesIO
import requests
import fitz  # PyMuPDF for extracting text
import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.utils import *

# Google Drive public folder link
def extract_folder_id_from_link(drive_link: str) -> str:
    """Extracts folder ID from a Google Drive link."""
    folder_id = drive_link.split('/')[-1].split('?')[0]
    return folder_id

def get_public_pdfs_from_drive(drive_link: str):
    """Fetches PDF files from a public Google Drive folder without OAuth authentication."""
    folder_id = extract_folder_id_from_link(drive_link)
    # Google Drive API URL to list files
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}' in parents and mimeType='application/pdf'"
    
    # Make a request to Google Drive API to get the file list
    response = requests.get(url)
    files = response.json().get('files', [])
    
    pdf_texts = {}
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        
        # Fetch the file content
        file_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        file_content = requests.get(file_url).content
        
        # Process PDF content
        pdf_stream = BytesIO(file_content)
        text = extract_text_from_pdf(pdf_stream)
        pdf_texts[file_name] = text

    return pdf_texts

def extract_text_from_pdf(pdf_stream):
    """Extracts text from an in-memory PDF."""
    text = ""
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def clean_and_structure_jd(jd_text, openai_api_key):
    """Uses OpenAI to clean and structure JD text into a detailed JSON format."""
    agent = Agent(model=OpenAIChat(model="gpt-4", api_key=openai_api_key))
    prompt = f"""
    Given the following job description text, extract and structure it into a JSON format with the following keys:
    - Job Role
    - Required Skills
    - Responsibilities
    - Qualifications
    - Experience Required
    - Company Name (if available)
    
    Job Description:
    {jd_text}
    
    Return a valid JSON format.
    """
    response = agent.chat(prompt)
    return json.loads(response)

def process_jd_from_drive(drive_link: str, openai_api_key: str):
    """Fetch, process, and store structured job descriptions from Drive PDFs."""
    pdf_texts = get_public_pdfs_from_drive(drive_link)  # Use public Google Drive link here
    structured_data = {}
    
    for filename, jd_text in pdf_texts.items():
        structured_data[filename] = clean_and_structure_jd(jd_text, openai_api_key)
    
    # Save structured data into a JSON file
    with open("roles.json", "w") as f:
        json.dump(structured_data, f, indent=4)
    
    return structured_data

def init_session_state() -> None:
    """Initialize only necessary session state variables."""
    defaults = {
        'openai_api_key': "", 'drivelink': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main() -> None:
    st.title("AI Recruitment System")

    init_session_state()
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI Configuration
        st.subheader("OpenAI Settings")
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key, help="Get your API key from platform.openai.com")
        if api_key: st.session_state.openai_api_key = api_key
        link = st.text_input("Drive Link", value=st.session_state.drivelink, help="Paste Link to Drive containing JOB Desc.")
        if link : st.session_state.drivelink = link

        required_configs = {'OpenAI API Key': st.session_state.openai_api_key, 'Drive Link' : st.session_state.drivelink}

    missing_configs = [k for k, v in required_configs.items() if not v]
    if missing_configs:
        st.warning(f"Please configure the following in the sidebar: {', '.join(missing_configs)}")
        return

    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    drive_link = st.session_state.drivelink
    openai_api_key = st.session_state.openai_api_key

    # Process the job descriptions from Google Drive folder
    structured_data = process_jd_from_drive(drive_link, openai_api_key)

    # Display structured data
    st.subheader("Structured Job Descriptions")
    st.json(structured_data)

    # Reset button
    if st.sidebar.button("Reset Application"):
        for key in st.session_state.keys():
            if key != 'openai_api_key':
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
