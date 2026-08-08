from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

loader = PyPDFLoader("Complete_React_Notes_Full_Stack.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

split_docs = splitter.split_documents(docs)
print(f"Number of split documents: {len(split_docs)}")
print(f"Content of the first split document: {split_docs[0].page_content}")