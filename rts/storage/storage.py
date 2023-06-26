from supabase import create_client, Client
from rts.db_settings import SUPABASE_HOST, SUPABASE_KEY


def get_supabase_client() -> Client:
    print("Creating supabase client")
    return create_client(
        SUPABASE_HOST,
        SUPABASE_KEY,
    )
