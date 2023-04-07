from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import (API_PREFIX, APP_NAME, APP_VERSION,
                             IS_DEBUG)
from app.core.event_handlers import model_lifespan

origins = [
    "http://localhost",
    "http://localhost:4200",
]


def get_app() -> FastAPI:
    fast_app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=IS_DEBUG, lifespan=model_lifespan)
    fast_app.include_router(api_router, prefix=API_PREFIX)
    fast_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return fast_app


app = get_app()
