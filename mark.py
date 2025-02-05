import os
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from googleapiclient.discovery import build
from google.oauth2 import service_account
import mimetypes
import io
from googleapiclient.http import MediaIoBaseDownload  # Import for downloads
from googleapiclient.http import MediaFileUpload  # Correct import
# --- MongoDB Connection ---
uri = "mongodb+srv://setooproject00:pass@setooproject.tvovq.mongodb.net/?retryWrites=true&w=majority&appName=setooproject"
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client["your_database_name"]  # Replace with your database name
    jds_collection = db["jds"]  # Collection for JDs
    resumes_collection = db["resumes"]  # Collection for resumes
    print("Connected to MongoDB Atlas!")
except pymongo.errors.ConnectionFailure as e:
    print(f"Could not connect to MongoDB Atlas: {e}")
    exit()
except Exception as e:
    print(f"An error occurred with MongoDB: {e}")
    exit()

# --- Google Drive API Setup ---
SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account_credentials.json'
resumes_folder = "Resume_collection"

try:
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    print("Connected to Google Drive!")
except Exception as e:
    print(f"Error connecting to Google Drive API: {e}")
    exit()

# --- Google Drive Functions ---

def upload_to_drive(file_path, drive_folder_id): # Correct: 2 arguments
    """Uploads a file to Google Drive from a file path and returns the file ID."""
    try:
        mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        file_metadata = {'name': os.path.basename(file_path), 'parents': [drive_folder_id]}
        media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(F'File ID: {file.get("id")}')
        return file.get("id")
    except Exception as e:
        print(f"Error uploading to Google Drive: {e}")
        return None
    
def store_metadata(original_filename, drive_file_id, collection, drive_folder_id):  # The missing function!
    """Stores file metadata in MongoDB."""
    try:
        metadata = {
            "original_filename": original_filename,
            "drive_file_id": drive_file_id,
            "drive_folder_id": drive_folder_id,
        }
        collection.insert_one(metadata)
        return True
    except Exception as e:
        print(f"Error storing metadata in MongoDB: {e}")
        return False

def fetch_file_content_from_drive(file_id):
    """Fetches file content from Google Drive given the file ID."""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print("Download %d%%." % int(status.progress() * 100))  # Optional progress
        fh.seek(0)
        return fh.getvalue()
    except Exception as e:
        print(f"Error fetching file from Google Drive: {e}")
        return None


# --- Test Upload Functions ---

def test_upload_files_to_drive(folder_path, drive_folder_id, collection): # Correct Definition
    """Test function to upload all files in a folder to Drive."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                drive_file_id = upload_to_drive(file_path, drive_folder_id) # Correct Call
                if drive_file_id:
                    if store_metadata(filename, drive_file_id, collection, drive_folder_id):
                        print(f"File '{filename}' uploaded and metadata saved.")
                    else:
                        print(f"Error saving metadata for '{filename}'.")
                else:
                    print(f"File '{filename}' upload failed.")
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

# --- Main Execution ---

# Replace with your actual folder IDs
JD_DRIVE_FOLDER_ID = "1sgBoF95YAHVfIlMFqPX76yD846QsQcFe"  # Correct JD folder ID
RESUME_DRIVE_FOLDER_ID = "1WaJPawJ55Hy4Z0f7-H1onZ22k077vURO"  # Correct resume folder ID

# Test Upload JDs
test_upload_files_to_drive("Job_descriptions", JD_DRIVE_FOLDER_ID, jds_collection)

# Test Upload Resumes
test_upload_files_to_drive("Resume_collection", RESUME_DRIVE_FOLDER_ID, resumes_collection)

print("File uploads and metadata storage test completed.")