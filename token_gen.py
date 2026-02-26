import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

api_key = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")

kite = KiteConnect(api_key=api_key)

print("\n1) Open this URL in browser and login:")
print(kite.login_url())

print("\n2) After login, you will be redirected to your Redirect URL.")
print("   Copy the value of request_token from the browser URL and paste it here.\n")

request_token = input("Enter request_token: ").strip()

data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

print("\n✅ Access token generated:")
print(access_token)

print("\nTip: you can store it in .env for today, or copy-paste into the app config.")
