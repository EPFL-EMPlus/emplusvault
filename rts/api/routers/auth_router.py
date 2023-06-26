from fastapi import FastAPI, HTTPException, Depends, Form, APIRouter
from rts.db_settings import DATABASE_URL, DB_HOST, DB_NAME, SUPABASE_HOST, SUPABASE_KEY
from supabase import create_client, Client
from rts.storage.storage import get_supabase_client
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
auth_router = APIRouter()


@auth_router.post("/login", include_in_schema=False)
async def login(username: str = Form(...), password: str = Form(...)):

    try:
        response = get_supabase_client().auth.sign_in_with_password(
            {"email": username, "password": password})
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Error logging in")
    return response.session


async def authenticate(token: str = Depends(oauth2_scheme)):
    supabase = create_client(
        SUPABASE_HOST,
        SUPABASE_KEY,
    )
    # res = supabase.auth.set_session(access_token=token, refresh_token=token)
    pgclient = supabase.postgrest
    pgclient.auth(token)
    return supabase


# @auth_router.get("/users/me/")
# async def read_users_me(supabase: Client = Depends(authenticate)):
#     return {"success": True}


# @auth_router.get("/read/media/")
# async def read_media_files(supabase: Client = Depends(authenticate)):
#     # return supabase.table('media').select("*").execute()
#     from rts.db.queries import read_media
#     return read_media()
