
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://setooproject00:pass@setooproject.tvovq.mongodb.net/?retryWrites=true&w=majority&appName=setooproject"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["your_database_name"]  # Replace with your database name
test_collection = db["test_data"]
print("Connected to MongoDB Atlas!")

# 2. Store Data (Example)
data_to_store = "This is some test data."
try:
    test_collection.insert_one({"data": data_to_store})
    print(f"Data '{data_to_store}' stored in MongoDB.")
except Exception as e:
    print(f"Error storing data: {e}")

# 3. Retrieve Data (Example)
try:
    retrieved_data = list(test_collection.find())
    if retrieved_data:
        print("Data retrieved from MongoDB:")
        for item in retrieved_data:
            print(item["data"])
    else:
        print("No data found in the collection.")
except Exception as e:
    print(f"Error retrieving data: {e}")

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)