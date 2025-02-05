import os
import json
import fitz  # For PDF processing
import streamlit as st
import openai
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from googleapiclient.discovery import build
from google.oauth2 import service_account
import mimetypes
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload

# --- MongoDB Connection ---
MONGO_URI = "mongodb+srv://setooproject00:pass@setooproject.tvovq.mongodb.net/?retryWrites=true&w=majority&appName=setooproject"  # Set this environment variable
if not MONGO_URI:
    st.error("MONGO_URI environment variable not set.")
    st.stop()  # Stop Streamlit execution

try:
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["your_database_name"]  # Replace with your database name
    jds_collection = db["jds"]  # Collection for JDs (if using MongoDB for JDs too)
    resumes_collection = db["resumes"]  # Collection for resumes
    print("Connected to MongoDB Atlas!")
except pymongo.errors.ConnectionFailure as e:
    st.error(f"Could not connect to MongoDB Atlas: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred with MongoDB: {e}")
    st.stop()
# --- Google Drive API Setup ---
SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account_credentials.json'

try:
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    print("Connected to Google Drive!")
except Exception as e:
    st.error(f"Error connecting to Google Drive API: {e}")
    st.stop()

# --- Google Drive Functions ---
def fetch_file_content_from_drive(file_id):
    """Fetches file content from Google Drive given the file ID."""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()  # Use BytesIO to store in memory
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print("Download %d%%." % int(status.progress() * 100))  # Optional progress
        fh.seek(0)  # Go back to the beginning of the stream
        return fh.getvalue()  # Get the file content as bytes
    except Exception as e:
        st.error(f"Error fetching file from Google Drive: {e}") # Display the error in streamlit
        print(f"Error fetching file from Google Drive: {e}") # print the error in console
        return None
    
def process_resumes_and_match(roles_data):  # No resumes_folder argument
    matched_resumes = {}
    unmatched_resumes = []

    for role_name, role_data in roles_data.items():
        matched_resumes[role_name] = []

    for resume_metadata in resumes_collection.find():  # Iterate through MongoDB
        filename = resume_metadata["original_filename"]
        drive_file_id = resume_metadata["drive_file_id"]
        try:
            resume_content = fetch_file_content_from_drive(drive_file_id)
            if resume_content:
                resume_text = extract_text_from_pdf_memory(resume_content, filename) # Extract from memory
                if resume_text:
                    best_match_role = None
                    best_match_score = 0

                    for role_name, role_data in roles_data.items():
                        # ... (The rest of the matching logic remains the same, using resume_text)
                        resume_words = set(resume_text.lower().split())
                        role_words = set()
                        if isinstance(role_data, list):
                            for role_item in role_data:
                                for value in role_item.values():
                                    if isinstance(value, list):
                                        for item in value:
                                            role_words.update(str(item).lower().split())
                                    elif isinstance(value, dict):
                                        for item in value.values():
                                            role_words.update(str(item).lower().split())
                                    elif value is not None:
                                        role_words.update(str(value).lower().split())
                        elif isinstance(role_data, dict):
                            for value in role_data.values():
                                if isinstance(value, list):
                                    for item in value:
                                        role_words.update(str(item).lower().split())
                                elif isinstance(value, dict):
                                    for item in value.values():
                                        role_words.update(str(item).lower().split())
                                elif value is not None:
                                    role_words.update(str(value).lower().split())

                        common_words = resume_words.intersection(role_words)

                        if common_words:  # Only proceed if there are common words
                            similarity_score = compare_resume_to_role(resume_text, role_data)
                            if 0.40 <= similarity_score <= 1 and similarity_score > best_match_score:
                                best_match_role = role_name
                                best_match_score = similarity_score

                    if best_match_role:
                        matched_resumes[best_match_role].append({
                            "resume_filename": filename,
                            "similarity_score": best_match_score
                        })
                    else:
                        unmatched_resumes.append(filename)
                else:
                    st.error(f"Could not extract from {filename} (Drive ID: {drive_file_id})")

            else:
                st.error(f"Could not download {filename} from Drive (ID: {drive_file_id})")


        except Exception as e:
            st.error(f"Error processing resume {filename}: {e}")

    return matched_resumes, unmatched_resumes

def extract_text_from_pdf_memory(pdf_bytes, filename): # Modified to accept bytes
    """Extracts text from PDF content in memory."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Open from memory
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Error extracting text from {filename}: {e}")
        return None
    return text

def compare_resume_to_role(resume_text, role_data):
    """Compares a resume to a role description (handles list of dicts)."""
    role_text = ""

    if isinstance(role_data, list):  # Check if role_data is a list
        for role_item in role_data:  # Iterate through list items (dicts)
            for value in role_item.values(): # Iterate through values within each dict
                if isinstance(value, list):
                    role_text += " ".join(str(item) for item in value) + " "
                elif isinstance(value, dict):
                    role_text += " ".join(str(item) for item in value.values()) + " "
                elif value is not None:
                    role_text += str(value) + " "
    elif isinstance(role_data, dict): # Handle if role_data is a dictionary
        for value in role_data.values():
            if isinstance(value, list):  # Check if the value is a list
                role_text += " ".join(str(item) for item in value) + " "
            elif isinstance(value, dict): # Check if the value is a dictionary
                role_text += " ".join(str(item) for item in value.values()) + " "
            elif value is not None:
                role_text += str(value) + " "

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, role_text])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def process_resumes_and_match(roles_data):  # No resumes_folder argument
    """Processes resumes from MongoDB/Drive and matches them to roles."""
    matched_resumes = {}
    unmatched_resumes = []

    for role_name, role_data in roles_data.items():
        matched_resumes[role_name] = []

    for resume_metadata in resumes_collection.find():  # Iterate through MongoDB
        filename = resume_metadata.get("original_filename")  # Handle missing filename
        drive_file_id = resume_metadata.get("drive_file_id")  # Handle missing drive_file_id

        if not filename or not drive_file_id:  # Skip if metadata is incomplete
            st.warning(f"Incomplete resume metadata in MongoDB: {resume_metadata}")
            continue  # Go to the next resume

        try:
            resume_content = fetch_file_content_from_drive(drive_file_id)
            if resume_content:
                resume_text = extract_text_from_pdf_memory(resume_content, filename)
                if resume_text:
                    best_match_role = None
                    best_match_score = 0

                    for role_name, role_data in roles_data.items():
                        similarity_score = compare_resume_to_role(resume_text, role_data)
                        if 0.40 <= similarity_score <= 1 and similarity_score > best_match_score:
                            best_match_role = role_name
                            best_match_score = similarity_score

                    if best_match_role:
                        matched_resumes[best_match_role].append({
                            "resume_filename": filename,
                            "similarity_score": best_match_score
                        })
                    else:
                        unmatched_resumes.append(filename)
                else:
                    st.error(f"Could not extract text from {filename} (Drive ID: {drive_file_id})")
            else:
                st.error(f"Could not download {filename} from Drive (ID: {drive_file_id})")

        except Exception as e:
            st.error(f"Error processing resume {filename}: {e}")

    return matched_resumes, unmatched_resumes

def generate_analytics(matched_resumes):
    """Generates analytics based on matched resumes."""
    analytics = {}
    for role_name, resumes in matched_resumes.items():
        analytics[role_name] = {
            "applied_count": len(resumes),
            "passed_count": sum(1 for r in resumes if r["similarity_score"] >= 0.5)
        }
    return analytics

def display_analytics(analytics):
    """Displays analytics, handling empty data gracefully."""
    st.subheader("Recruitment Analytics")

    # 1. Handle potentially empty analytics
    if not analytics:  # Check if the entire analytics dictionary is empty
        st.write("No analytics to display yet.")
        return  # Exit early if no analytics

    # 2. Prepare data for DataFrame (handling empty role data)
    data = {}
    for role, counts in analytics.items():
        if not counts: # Check if the count dictionary is empty for a role
            data[role] = {"applied_count": 0, "passed_count": 0}
        else:
            data[role] = counts

    # 3. Tabular Display
    st.write("### Analytics Table")
    df = pd.DataFrame.from_dict(data, orient='index')
    st.dataframe(df)

    # 4. Improved Bar Chart (Plotly)
    st.write("### Applications per Job Role")

    roles = list(data.keys())  # Use the potentially modified data
    applications = [data[role]["applied_count"] for role in roles]
    passed = [data[role].get("passed_count", 0) for role in roles] # Handle missing passed_count

    fig = go.Figure(data=[
        go.Bar(name='Applied', x=roles, y=applications, marker_color='skyblue'),
        go.Bar(name='Passed', x=roles, y=passed, marker_color='forestgreen')
    ])

    fig.update_layout(
        title="Applications per Job Role",
        xaxis_title="Job Role",
        yaxis_title="Number of Applications",
        barmode='group'
    )
    st.plotly_chart(fig)

    # 5. Summary Statistics
    st.write("### Summary Statistics")
    total_applications = sum(applications)
    st.write(f"Total Applications: {total_applications}")

    total_passed = sum(passed)
    if total_applications > 0:
        average_pass_rate = (total_passed / total_applications) * 100
        st.write(f"Average Pass Rate: {average_pass_rate:.2f}%")
    else:
        st.write("Average Pass Rate: N/A (No applications)")

def display_top_resumes(matched_resumes, role_name, top_n=5):
    if role_name not in matched_resumes:
        st.warning(f"No resumes found for role: {role_name}")
        return

    resumes_for_role = matched_resumes[role_name]
    sorted_resumes = sorted(resumes_for_role, key=lambda x: x['similarity_score'], reverse=True)

    st.subheader(f"Top Resumes for {role_name}")

    if not sorted_resumes:
        st.write("No resumes matched this role.")
        return

    num_resumes_to_display = min(top_n, len(sorted_resumes))
    st.write(f"Displaying top {num_resumes_to_display} resumes:")

    for i in range(num_resumes_to_display):
        resume = sorted_resumes[i]
        resume_filename = resume['resume_filename']
        st.markdown(f'<a href="#" onclick="return false;">- {resume_filename} (Similarity: {resume["similarity_score"]:.2f})</a>', unsafe_allow_html=True)

        resume_metadata = resumes_collection.find_one({"original_filename": resume_filename})
        if resume_metadata and "drive_file_id" in resume_metadata:  # Check if drive_file_id exists
            drive_file_id = resume_metadata["drive_file_id"]
            resume_content = fetch_file_content_from_drive(drive_file_id)  # Get from Drive
            if resume_content:
                base64_pdf = base64.b64encode(resume_content).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.error(f"Could not retrieve resume content from Drive for: {resume_filename}")
        else:
            st.error(f"Drive file ID not found for: {resume_filename}")  # Handle missing Drive ID
def extract_text_from_pdf(pdf_path):  # Corrected function signature
    """Extracts text from a PDF file (given its path)."""
    text = ""
    try:
        doc = fitz.open(pdf_path) # Open using the path
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")  # More informative error
        return None
    return text

def clean_and_structure_jd(jd_text, openai_api_key):
    """Uses OpenAI to clean and structure JD text."""
    openai.api_key = openai_api_key

    if not jd_text or not jd_text.strip():
        st.warning("Empty JD text, skipping structuring.")
        return {}

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

    Return a valid JSON format. If you cannot find a field, you can leave it as an empty string or null value. Do not explain your reasoning, just return the JSON.
    """

    try:
        response = openai.ChatCompletion.create(  # Use ChatCompletion.create
            model="gpt-4",  # Or gpt-3.5-turbo
            messages=[  # Use 'messages' for chat models
                {"role": "system", "content": "You are a helpful assistant that extracts information from job descriptions and returns it as structured JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        json_string = response['choices'][0]['message']['content'].strip()  # Access content from 'message'

        try:
            structured_data = json.loads(json_string)
            return structured_data
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON returned by OpenAI: {e}. Raw Response: {json_string}")
            return {}

    except Exception as e:
        st.error(f"Error with OpenAI API call: {e}")
        return {}

def main():
    st.title("AI Recruitment System")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    # Initialize session state variables
    if "structured_data" not in st.session_state:
        st.session_state.structured_data = {}
    if "matched_resumes" not in st.session_state:
        st.session_state.matched_resumes = {}
    if "unmatched_resumes" not in st.session_state:
        st.session_state.unmatched_resumes = []
    if "analytics" not in st.session_state:
        st.session_state.analytics = {}
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = None
    if "analysis_performed" not in st.session_state:
        st.session_state.analysis_performed = False

    if not st.session_state.analysis_performed:
        if st.button("Process and Analyze"):
            # JD Processing (from MongoDB and Drive)
            st.session_state.structured_data = {}
            for jd_metadata in jds_collection.find():
                filename = jd_metadata.get("original_filename")
                drive_file_id = jd_metadata.get("drive_file_id")

                if not filename or not drive_file_id:
                    st.warning(f"Incomplete JD metadata in MongoDB: {jd_metadata}")
                    continue

                try:
                    jd_content = fetch_file_content_from_drive(drive_file_id)
                    if jd_content:
                        jd_text = extract_text_from_pdf_memory(jd_content, filename)
                        if jd_text:
                            st.session_state.structured_data[filename] = clean_and_structure_jd(jd_text, openai_api_key)
                        else:
                            st.error(f"No text extracted from {filename} (Drive ID: {drive_file_id})")
                    else:
                        st.error(f"Could not download JD {filename} from Drive (ID: {drive_file_id})")
                except Exception as e:
                    st.error(f"Error processing JD {filename}: {e}")

            if st.session_state.structured_data:
                # Resume Processing (from MongoDB and Drive)
                st.session_state.matched_resumes, st.session_state.unmatched_resumes = process_resumes_and_match(st.session_state.structured_data)  # No resumes_folder argument

                if st.session_state.matched_resumes:
                    st.session_state.analytics = generate_analytics(st.session_state.matched_resumes)
                    st.session_state.analysis_performed = True
                else:
                    st.warning("No resumes were successfully processed or matched.")
            else:
                st.warning("No job descriptions were successfully processed.")

    if st.session_state.analysis_performed:
        display_analytics(st.session_state.analytics)

        available_roles = list(st.session_state.matched_resumes.keys())

        def update_selected_role():
            st.session_state.selected_role = st.session_state.temp_selected_role

        st.session_state.temp_selected_role = st.selectbox(
            "Select a role to see top resumes:",
            available_roles,
            index=available_roles.index(st.session_state.selected_role) if st.session_state.selected_role in available_roles else 0,
            on_change=update_selected_role,
        )
        top_n = st.number_input("Number of top resumes to display:", min_value=1, value=5, step=1)

        if st.session_state.selected_role:
            display_top_resumes(st.session_state.matched_resumes, st.session_state.selected_role, top_n)

        if st.session_state.unmatched_resumes:
            st.subheader("Unmatched Resumes")
            for resume in st.session_state.unmatched_resumes:
                st.write(f"- {resume}")

if __name__ == "__main__":
    main()