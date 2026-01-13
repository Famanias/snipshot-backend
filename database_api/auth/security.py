"""
Supabase Authentication utilities

Verifies Supabase Auth JWT tokens using the Supabase client.
"""

import os
from typing import Optional
import jwt
from dotenv import load_dotenv

load_dotenv()

# Supabase JWT settings
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")


def decode_supabase_token(token: str) -> Optional[dict]:
    """
    Decode and verify a Supabase access token.
    Returns the payload if valid, None otherwise.
    
    Supabase JWTs can use either HS256 or ES256 depending on project settings.
    We try multiple approaches for compatibility.
    """
    # First, try to decode without verification to check the algorithm
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        return None
    
    # Get the algorithm from the token header
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg", "HS256")
    except jwt.InvalidTokenError:
        algorithm = "HS256"
    
    # For HS256, use the JWT secret
    if algorithm == "HS256" and SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated"
            )
            return payload
        except jwt.InvalidTokenError:
            pass
    
    # For ES256 or if HS256 failed, verify using Supabase client
    # The Supabase client validates the token against the auth server
    try:
        from config import get_supabase
        supabase = get_supabase()
        
        # Use Supabase to get user from token - this validates the token
        user_response = supabase.auth.get_user(token)
        
        if user_response and user_response.user:
            # Return a payload-like dict with essential info
            return {
                "sub": user_response.user.id,
                "email": user_response.user.email,
                "aud": "authenticated",
                "role": "authenticated"
            }
    except Exception as e:
        print(f"Token verification failed: {e}")
        pass
    
    return None


def get_user_id_from_token(token: str) -> Optional[str]:
    """Extract user ID from Supabase token"""
    payload = decode_supabase_token(token)
    if payload:
        return payload.get("sub")  # Supabase uses 'sub' for user ID
    return None

