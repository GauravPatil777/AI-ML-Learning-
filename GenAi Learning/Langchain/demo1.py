from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
result=model.invoke("Write a poem about the beauty of nature in 5 lines.")
print(result.text)