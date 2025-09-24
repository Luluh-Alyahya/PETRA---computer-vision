from pydantic import BaseModel, HttpUrl

class ImageURL(BaseModel):
    url: HttpUrl
