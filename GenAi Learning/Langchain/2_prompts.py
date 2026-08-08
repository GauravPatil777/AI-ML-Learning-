from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

prompt = PromptTemplate(
    template="Explain {topic} in 5 lines.",
    input_variables=["topic"]
)
chain=prompt | model
result=chain.invoke({"topic": "RAG"})
print(result.text)