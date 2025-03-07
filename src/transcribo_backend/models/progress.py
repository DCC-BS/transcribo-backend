from pydantic import BaseModel


class ProgressResponse(BaseModel):
    progress: float
    currentTime: float
    duration: float
