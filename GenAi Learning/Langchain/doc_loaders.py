from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"  
)

# # text loader
# loader=TextLoader("data.txt")
# docs=loader.load()
# print(docs)

# web loader
# from langchain_community.document_loaders import WebBaseLoader

# web_loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/document_loaders/unstructured_file")
# web_docs = web_loader.load()
# print(web_docs[0].page_content)

from langchain_community.document_loaders import PyPDFLoader

pdf_loader=PyPDFLoader("Complete_React_Notes_Full_Stack.pdf")
docs=pdf_loader.load()
print(len(docs))
print(docs[0].page_content)