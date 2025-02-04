import os
import json
import fitz
import streamlit as st
import openai
import os
import json
import fitz
import os
import json
import fitz
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import base64
import streamlit as st
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import json
import fitz
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import base64

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

def process_resumes_and_match(roles_data, resumes_folder):
    """Processes resumes, matches them to roles, and assigns them based on similarity."""
    matched_resumes = {}
    unmatched_resumes = []

    try:
        for role_name, role_data in roles_data.items():
            matched_resumes[role_name] = []

        for filename in os.listdir(resumes_folder):
            if filename.endswith(".pdf"):
                resume_path = os.path.join(resumes_folder, filename)
                try:
                    resume_text = extract_text_from_pdf(resume_path)
                    if resume_text:
                        best_match_role = None
                        best_match_score = 0

                        for role_name, role_data in roles_data.items():
                            # Check for keyword overlap first (at least one common word)
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
                        st.error(f"No text extracted from {filename}")
                except Exception as e:
                    st.error(f"Error processing resume {filename}: {e}")

    except FileNotFoundError:
        st.error(f"Resumes folder '{resumes_folder}' not found.")
        return {}, []

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

def display_top_resumes(matched_resumes, role_name, top_n=5):  # Add top_n parameter
    """Displays top N resumes with clickable links."""
    if role_name not in matched_resumes:
        st.warning(f"No resumes found for role: {role_name}")
        return

    resumes_for_role = matched_resumes[role_name]
    sorted_resumes = sorted(resumes_for_role, key=lambda x: x['similarity_score'], reverse=True)

    st.subheader(f"Top Resumes for {role_name}")

    if not sorted_resumes:  # Check if the list is empty
        st.write("No resumes matched this role.")
        return

    num_resumes_to_display = min(top_n, len(sorted_resumes))  # Display at most top_n or all
    st.write(f"Displaying top {num_resumes_to_display} resumes:")

    resumes_folder = "Resume_collection"  # Correct path to your resumes

    for i in range(num_resumes_to_display):  # Iterate only up to the limit
        resume = sorted_resumes[i]
        resume_filename = resume['resume_filename']
        resume_path = os.path.join(resumes_folder, resume_filename)

        if os.path.exists(resume_path):
            st.write(f"- {resume_filename} (Similarity: {resume['similarity_score']:.2f})")

            with open(resume_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)  # CAUTION: See previous explanation

        else:
            st.error(f"Resume file not found: {resume_path}")

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
    """Uses OpenAI to clean and structure JD text and *extract the Job Role*."""
    openai.api_key = openai_api_key

    if not jd_text or not jd_text.strip():
        st.warning("Empty JD text, skipping structuring.")
        return {}

    prompt = f"""
    Given the following job description text, extract and structure it into a JSON format.  **Crucially, identify and extract the actual *Job Role* from the description.**

    Job Description:
    {jd_text}

    Return a valid JSON format with the following keys (including Job Role):
    - Job Role  (The actual job role, e.g., "Software Engineer," not the PDF filename)
    - Required Skills
    - Responsibilities
    - Qualifications
    - Experience Required
    - Company Name (if available)

    If you cannot find a field, you can leave it as an empty string or null value. Do not explain your reasoning, just return the JSON.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from job descriptions and returns it as structured JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        json_string = response['choices'][0]['message']['content'].strip()

        try:
            structured_data = json.loads(json_string)
            job_role = structured_data.get("Job Role")  # Extract the Job Role
            if not job_role:
                st.error("Could not extract Job Role from JD.")
                return {}
            return {job_role: structured_data}  # Use Job Role as the key

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

    folder_path = "Job_descriptions"  # Hardcoded folder name

    resumes_folder = "Resume_collection"

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

    if not st.session_state.analysis_performed:  # Only show button if analysis not yet done
        if st.button("Process and Analyze"):
            if os.path.exists(folder_path) and os.path.exists(resumes_folder):
                # JD Processing
                st.session_state.structured_data = {}  # Clear old JD data
                for filename in os.listdir(folder_path):
                    if filename.endswith(".pdf"):
                        pdf_path = os.path.join(folder_path, filename)
                        try:
                            with open(pdf_path, "rb") as f:
                                jd_text = extract_text_from_pdf(f)
                                if jd_text:
                                    st.session_state.structured_data[filename] = clean_and_structure_jd(jd_text, openai_api_key)
                                else:
                                    st.error(f"No text extracted from {filename}")
                        except Exception as e:
                            st.error(f"Error processing JD {filename}: {e}")

                if st.session_state.structured_data:
                    st.session_state.matched_resumes, st.session_state.unmatched_resumes = process_resumes_and_match(
                        st.session_state.structured_data, resumes_folder
                    )

                    if st.session_state.matched_resumes:
                        st.session_state.analytics = generate_analytics(st.session_state.matched_resumes)
                        st.session_state.analysis_performed = True  # Set to True after analysis
                    else:
                        st.warning("No resumes were successfully processed.")  # Or no matches found
                else:
                    st.warning("No job descriptions were successfully processed.")
            else:
                if not os.path.exists(folder_path):
                    st.error(f"Folder '{folder_path}' does not exist.")
                if not os.path.exists(resumes_folder):
                    st.error(f"Folder '{resumes_folder}' does not exist.")

    if st.session_state.analysis_performed:  # Show analytics only if analysis is done
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

        top_n = st.number_input("Number of top resumes to display:", min_value=1, value=5, step=1)  # Default 5

        if st.session_state.selected_role:
            display_top_resumes(st.session_state.matched_resumes, st.session_state.selected_role, top_n)  # Pass top_n
        if st.session_state.unmatched_resumes:  # Display unmatched resumes
            st.subheader("Unmatched Resumes")
            for resume in st.session_state.unmatched_resumes:
                st.write(f"- {resume}")


if __name__ == "__main__":
    main()