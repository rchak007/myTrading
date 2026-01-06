import os
import dotenv
from schwabdev import Client

# Load your lowercase .env
dotenv.load_dotenv()

app_key = os.getenv("app_key")
app_secret = os.getenv("app_secret")
callback_url = os.getenv("callback_url")

client = Client(app_key, app_secret, callback_url)

# Print all methods that contain 'trans' in the name
methods = [m for m in dir(client) if "trans" in m.lower()]
print("\nTransaction-related methods on Schwabdev Client:\n")
for m in methods:
    print(" -", m)

