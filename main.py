from dotenv import load_dotenv
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectore_store = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriver = vectore_store.as_retriever(
    search_type = "mmr",
    search_kwargs= {
        "k":2,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

promt = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a helpful assistent. Use ONLY the provided context to answer the question. It the answer is not present in the context, sya: "I could not find the answer in the document." """),
        ("human",""" Context: {context}
            Questions: {question}
         """)
    ]
)

print("Rag System Created. \nPress 0 to exit ")

while True:
    query= input("You: ")
    if query == "0": break

    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_promt = promt.invoke({
        "context": context,
        "question": query
    })


    response = llm.invoke(final_promt)

    print(f"\nAI: {response.content}")

