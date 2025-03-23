import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import NewsAnalysis
from fastapi.responses import FileResponse

app =  FastAPI()

class NewsRequest(BaseModel):
    company_name: str

@app.get("/")
def home():
    return {"message": "News Analysis is Running!"}

@app.post("/fetch_news")
def fetch_news(request: NewsRequest):
    """Fetch news articles, analyze sentiment, and return everything including audio."""
    try:
        analysis = NewsAnalysis(request.company_name)
        result  = analysis.main()

        audio_file = result['report'].get("Audio",None)

        return {
            "articles": result["articles"],
            "report": result["report"],
            "audio_file": audio_file if audio_file else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-audio")
def download_audio():
    """Returns the generated Hindi audio file."""
    audio_path = "summary.mp3"
    try:
        return FileResponse("summary.mp3", media_type="audio/mp3", filename="summary.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)