from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator
from pydantic_core import PydanticCustomError

from app.services.names import clean_full_name


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(max_length=100)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        try:
            return clean_full_name(value)
        except ValueError as exc:
            # PydanticCustomError: arayüzde mesajın başına "Value error," eklenmesin.
            raise PydanticCustomError("full_name", str(exc)) from exc


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    full_name: str | None = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str | None = None

    class Config:
        from_attributes = True