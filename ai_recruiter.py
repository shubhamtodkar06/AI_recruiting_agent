import os
import json
import fitz
import streamlit as st
import openai

def extract_text_from_pdf(pdf_stream):
    """Extracts text from an in-memory PDF."""
    text = ""
    try:
        pdf_bytes = pdf_stream.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
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

    if st.button("Process Job Descriptions"):  # Button to start processing
        if os.path.exists(folder_path):
            structured_data = {}
            for filename in os.listdir(folder_path):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(folder_path, filename)
                    try:  # Handle file open errors
                        with open(pdf_path, "rb") as f:  # Open file here for each PDF
                            jd_text = extract_text_from_pdf(f)
                            if jd_text:
                                structured_data[filename] = clean_and_structure_jd(jd_text, openai_api_key)
                            else:
                                st.error(f"No text extracted from {filename}")
                    except Exception as e:
                        st.error(f"Error opening file {filename}: {e}")


            st.subheader("Structured Job Descriptions")

            if structured_data:
                st.write("Saving to roles.json...")
                try:
                    with open("roles.json", "w", encoding="utf-8") as f:
                        json.dump(structured_data, f, indent=4, ensure_ascii=False)
                    st.success("Saved to roles.json")

                    for filename, data in structured_data.items():
                        st.subheader(filename)
                        if data:
                            st.json(data)  # Or st.write(data) or st.dataframe(data)
                        else:
                            st.write("No structured data extracted.")

                except Exception as e:
                    st.error(f"Error saving to roles.json: {e}")
            else:
                st.warning("No job descriptions were successfully processed.")
        else:
            st.error(f"Folder '{folder_path}' does not exist.")

if __name__ == "__main__":
    main()