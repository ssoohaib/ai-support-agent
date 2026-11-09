# app/auth.py
import time
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import settings

bearer = HTTPBearer()

class User(BaseModel):
    id: str
    role: str

def issue_token(user_id: str, role: str) -> str:
    claims = {"sub": user_id, "role": role, "exp": int(time.time()) + 30 * 60}
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")

def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> User:
    try:
        claims = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    return User(id=claims["sub"], role=claims["role"])

def require_role(*roles: str):
    def check(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "Forbidden")
        return user
    return check