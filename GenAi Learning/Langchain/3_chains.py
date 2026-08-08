from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

# 1. Large text
text = """
Artificial intelligence is a field of computer science that focuses on
creating systems capable of performing tasks that normally require human
intelligence. These tasks include learning, reasoning, problem solving,
understanding natural language, recognizing images, and making decisions.

Machine learning is a major part of artificial intelligence. Instead of
being explicitly programmed for every task, machine learning systems learn
patterns from data. Deep learning is a subset of machine learning that uses
neural networks with multiple layers to learn complex patterns.

Large language models are deep learning models trained on huge amounts of
text. They can understand and generate human-like text. Models such as
modern generative AI systems can be used for summarization, question
answering, translation, coding, and many other tasks.
"""
# prompt template for summarization
prompt=PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
)

# Output parser
parser = StrOutputParser()
# sequential chain
chain= prompt | model | parser

chain_result=chain.invoke({"text":text})
print(chain_result)

from langchain_core.runnables import RunnableParallel
# parallel chain

# summary prompt
summary_prompt=PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
)
# summary chain
summary_chain= summary_prompt | model | parser

# question prompt
question_prompt=PromptTemplate(
    template="Generate 5 questions answers from following text:\n{text}",
    input_variables=["text"]
)
# question chain
question_chain= question_prompt | model | parser

# parallel chain
 
parallel_chain= RunnableParallel(
    summary=summary_chain, 
    qna=question_chain
) 
result=parallel_chain.invoke({"text":text})
print("SUMMARY: ",result)

print("\nQUESTIONS & ANSWERS")
print(result["qna"])