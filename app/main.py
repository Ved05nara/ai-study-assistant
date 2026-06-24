from fastapi import FastAPI
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router
from app.services.pdf_service import extract_text_from_pdf
app = FastAPI()

app.include_router(upload_router)
app.include_router(query_router)

@app.get("/")
def root():
    return {
        "message": "AI Study Assistant Running"
    }