from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated,TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)
# TypedDict for structured output
# class Movie(TypedDict):
#     title:Annotated[str, "The title of the movie"]
#     year:Annotated[int, "The year the movie was released"]
#     director:Annotated[str, "The director of the movie"]
#     collection:Annotated[str, "The box office collection of the movie in crores"]



# Pydantic model for structured output
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title:str=Field(..., description="The title of the movie")
    year:int=Field(..., description="The year the movie was released")
    director:str=Field(..., description="The director of the movie")
    collection:str=Field(..., description="The box office collection of the movie in crores")

     
structured_model=model.with_structured_output(Movie)

result=structured_model.invoke("Provide details of the movie RRR.")
print(result)

# json schema for structured output