from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# video transcript fetching
video_id="NEUrWO3496w"
api=YouTubeTranscriptApi()
available_transcripts = api.list(video_id)

selected_transcript = None

for transcript in available_transcripts:
    if selected_transcript is None:
        selected_transcript = transcript

transcript=api.fetch(video_id,languages=[selected_transcript.language_code])
full_transcript=" ".join(chunk.text for chunk in transcript)



# for transcript in transcripts:
#     print(
#         transcript.language,
#         transcript.language_code,
#         transcript.is_generated
#     )

#text splitting
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
transcripted_chunks=text_splitter.split_text(full_transcript)
# print(f"Total chunks created: {len(transcripted_chunks)}")
# print(f"First chunk: {transcripted_chunks[0]}")

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
# vector store creation
vector_store=Chroma.from_texts(
    transcripted_chunks,
    embedding=embeddings
)
# user question
user_question="which improvements can i do later in my project?"

# similarity search
context=vector_store.similarity_search(user_question, k=3)
# print(f"Context retrieved: {context}")
prompt=PromptTemplate(
    input_variables=["context","question"],
    template="You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
)

model=ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)
parser=StrOutputParser()
chain=prompt | model | parser
result=chain.invoke({"context":" ".join([doc.page_content for doc in context]),"question":user_question})
print(result)
