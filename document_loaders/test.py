from langchain_community.document_loaders import TextLoader



data = TextLoader("document_loaders/note.txt")

# print(data)

docs = data.load()

# print(docs[0].metadata)
print(docs[0].page_content)