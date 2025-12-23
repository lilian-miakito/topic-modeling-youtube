"""
To-Peek Backend - FastAPI Application

Main entry point for the topic modeling backend API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import datasets_router, topics_router, channels_router

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for YouTube comment topic modeling and analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(datasets_router, prefix=settings.API_V1_PREFIX)
app.include_router(topics_router, prefix=settings.API_V1_PREFIX)
app.include_router(channels_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.PROJECT_NAME,
        "version": "0.1.0",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

