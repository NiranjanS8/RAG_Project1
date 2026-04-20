from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Machine learning models require large datasets for training.",
        metadata={"source": "ML_Guide"}
    ),
    Document(
        page_content="Pandas is a powerful data manipulation library in Python.",
        metadata={"source": "Data_science"}
    ),
    Document(
        page_content="Deep learning uses neural networks with multiple layers.",
        metadata={"source": "DL_Tutorial"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

# print("Chroma DB created successfully.")

result = vector_store.similarity_search("What is used for data analysis?", k= 2)

for r in result:
    print(r)

retriver = vector_store.as_retriever()

docs = retriver.invoke("What is ML?")

for d in docs:
    print(d.page_content)