import os
import json
import fitz
import streamlit as st
import openai
import os
import json
import fitz
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
    unmatched_resumes = []  # List for resumes not matching any role

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
                            similarity_score = compare_resume_to_role(resume_text, role_data)
                            if 0.40 <= similarity_score <= 0.50 and similarity_score > best_match_score: # Check range and better match
                                best_match_role = role_name
                                best_match_score = similarity_score

                        if best_match_role:
                            matched_resumes[best_match_role].append({
                                "resume_filename": filename,
                                "similarity_score": best_match_score
                            })
                        else:
                            unmatched_resumes.append(filename) # Add to unmatched if no match
                    else:
                        st.error(f"No text extracted from {filename}")
                except Exception as e:
                    st.error(f"Error processing resume {filename}: {e}")

    except FileNotFoundError:
        st.error(f"Resumes folder '{resumes_folder}' not found.")
        return {}, []  # Return empty dict and list if folder not found

    return matched_resumes, unmatched_resumes  # Return both matched and unmatched

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
    """Displays analytics using Streamlit and visualizations."""
    st.subheader("Recruitment Analytics")

    st.write("Analytics Table:")
    st.write(analytics)

    roles = list(analytics.keys())
    applications = [data["applied_count"] for data in analytics.values()]

    fig, ax = plt.subplots()
    ax.bar(roles, applications)
    ax.set_xlabel("Job Role")
    ax.set_ylabel("Number of Applications")
    ax.set_title("Applications per Job Role")
    st.pyplot(fig)

def display_top_resumes(matched_resumes, role_name):
    """Displays the top resumes for a given role."""
    if role_name not in matched_resumes:
        st.warning(f"No resumes found for role: {role_name}")
        return

    resumes_for_role = matched_resumes[role_name]
    sorted_resumes = sorted(resumes_for_role, key=lambda x: x['similarity_score'], reverse=True)

    st.subheader(f"Top Resumes for {role_name}")
    for resume in sorted_resumes:
        st.write(f"- {resume['resume_filename']} (Similarity: {resume['similarity_score']:.2f})")

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

    folder_path = "Job_descriptions"  # Hardcoded folder name

    resumes_folder = "Resume_collection"

    if st.button("Process and Analyze"):
        if os.path.exists(folder_path) and os.path.exists(resumes_folder):
            structured_data = {}
            for filename in os.listdir(folder_path):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(folder_path, filename)
                    try:
                        with open(pdf_path, "rb") as f:
                            jd_text = extract_text_from_pdf(f)
                            if jd_text:
                                structured_data[filename] = clean_and_structure_jd(jd_text, openai_api_key)
                            else:
                                st.error(f"No text extracted from {filename}")
                    except Exception as e:
                        st.error(f"Error processing JD {filename}: {e}")

            if structured_data:
                matched_resumes, unmatched_resumes = process_resumes_and_match(structured_data, resumes_folder)

                if matched_resumes:
                    analytics = generate_analytics(matched_resumes)
                    display_analytics(analytics)

                    role_selection = st.selectbox("Select a role to see top resumes:", list(matched_resumes.keys()))
                    display_top_resumes(matched_resumes, role_selection)

                    if unmatched_resumes:  # Display unmatched resumes
                        st.subheader("Unmatched Resumes")
                        for resume in unmatched_resumes:
                            st.write(f"- {resume}")
                else:
                    st.warning("No resumes were successfully processed.")  # Or no matches found
            else:
                st.warning("No job descriptions were successfully processed.")
        else:
            if not os.path.exists(folder_path):
                st.error(f"Folder '{folder_path}' does not exist.")
            if not os.path.exists(resumes_folder):
                st.error(f"Folder '{resumes_folder}' does not exist.")

if __name__ == "__main__":
    main()