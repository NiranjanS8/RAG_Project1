from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document_loaders/Spring-Notes.pdf")

docs = data.load()

# print(docs[55].page_content)
# print(len(docs))

templete = ChatPromptTemplate.from_messages(
    [("system", "You are an expert text summarizer. MAX LENGTH = 4-5lines"), 
     ("human","{data}")]
)

promt = templete.format_messages(data = docs[0].page_content)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

result = model.invoke(promt)

print(result.content)