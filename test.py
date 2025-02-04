import kagglehub

try:
    kagglehub.dataset_download("shubhamtodkar06/job-descrption")  # Use the correct name
    print("Download successful!")
except Exception as e:
    print(f"Error: {e}")