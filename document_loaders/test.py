from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap = 1
)

data = TextLoader("document_loaders/note.txt")

# print(data)

docs = data.load()

chunks = splitter.split_documents(docs)

# print(docs[0].metadata)
# print(docs[0].page_content)

print(len(chunks))
# print(chunks)

for i in chunks:
    print(i.page_content)
    print(" ")