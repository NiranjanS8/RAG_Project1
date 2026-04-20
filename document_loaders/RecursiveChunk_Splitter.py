from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document_loaders/Spring-Notes.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 2
)

chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks[0].page_content)