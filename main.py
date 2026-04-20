from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
data = TextLoader("document_loaders/note.txt")

docs = data.load() 

templete = ChatPromptTemplate.from_messages(
    [("system", "You are an expert text summarizer"), 
     ("human","{data}")]
)

promt = templete.format_messages(data = docs[0].page_content)

model = ChatMistralAI(model = "mistral-small-2506")

result = model.invoke(promt)

print(result.content)

