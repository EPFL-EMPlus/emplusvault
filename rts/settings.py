import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
