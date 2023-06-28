from fastapi import FastAPI, HTTPException, Depends, Form, APIRouter
from rts.db_settings import DATABASE_URL, DB_HOST, DB_NAME, SUPABASE_HOST, SUPABASE_KEY
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
auth_router = APIRouter()


@auth_router.post("/login", include_in_schema=False)
async def login(username: str = Form(...), password: str = Form(...)):

    # try:
    #     response = get_storage_client().auth.sign_in_with_password(
    #         {"email": username, "password": password})
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=400, detail="Error logging in")
    # return response.session
    return "session"


async def authenticate(token: str = Depends(oauth2_scheme)):
    # supabase = create_client(
    #     SUPABASE_HOST,
    #     SUPABASE_KEY,
    # )
    # # res = supabase.auth.set_session(access_token=token, refresh_token=token)
    # pgclient = supabase.postgrest
    # pgclient.auth(token)
    # return supabase
    return "session"
