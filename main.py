from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
data = TextLoader("document_loaders/Spring-Notes.pdf")

docs = data.load() 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 2
)

templete = ChatPromptTemplate.from_messages(
    [("system", "You are an expert text summarizer"), 
     ("human","{data}")]
)

promt = templete.format_messages(data = docs[0].page_content)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

result = model.invoke(promt)

print(result.content)

