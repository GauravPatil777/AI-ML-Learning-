from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("Complete_React_Notes_Full_Stack.pdf")
docs = loader.load()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"

)
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
splitted_docs= splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    splitted_docs,
    embedding=embeddings
)
similar_vectors=vector_store.similarity_search("which hook used for state management?",1)

prompt=PromptTemplate(
    input_variables=["context","question"],
    template="Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
model=ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)
chain=prompt | model
result=chain.invoke({"context":similar_vectors[0].page_content,"question":"which hook used for state management?"})
print(result.text)
