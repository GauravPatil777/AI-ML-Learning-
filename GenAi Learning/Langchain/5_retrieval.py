from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)
documents = [
    "useState is a React Hook used to manage state.",
    "useEffect is used to perform side effects.",
    "useContext is used to consume context.",
    "useState is a very useful hook which is used to manage state.",
    "useRef is used to reference DOM elements."
]

vector_store = Chroma.from_texts(
    documents,
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
)

retriever = vector_store.as_retriever(
     search_kwargs={"k": 2},
     search_type="mmr"  # this prevents the retriever from returning similar documents and instead returns diverse documents based on the query.
)

# multiquery retrieval (this helps to get more relevant results for a query by using multiple queries)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model
)
results=multiquery_retriever.invoke("which hook used for state management?")
for doc in results:
    print(doc.page_content)
