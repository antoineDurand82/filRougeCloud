from fastapi import APIRouter
from app.api.routes import prediction

"""
Router of the API. It registers the routes and it prefix's. 
The openAPI documentation is automatically updated.
"""
api_router = APIRouter()


api_router.include_router(prediction.router, tags=["prediction"], prefix="/model")
