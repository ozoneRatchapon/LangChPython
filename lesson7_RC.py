# Retrieval Chain : load the document from website and split the document.
# load to word-embedding model and vector-stores.
# retrieve the answer from the vector-store.

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_together import TogetherEmbeddings




def  get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    splitDocs = splitter.split_documents(docs)
    # print(len(splitDocs))

    return splitDocs

def create_db(docs):
    # embedding = OpenAIEmbeddings()
    embedding = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
    )
    vectorStore = FAISS.from_documents(docs, embedding = embedding) 
    return vectorStore


def create_chain(vectorStore):
    # Initialize the ChatGroq object
    llm = ChatGroq(
        #model="llama-3.2-3b-preview",
        model ="llama-3.3-70b-versatile",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(""" 
        Answer the user's question:
        Context :  {context}
        User Question : {input}
    """)


    chain = create_stuff_documents_chain(
        llm =llm,
        prompt = prompt
        )
    
    retriever = vectorStore.as_retriever(search_kwargs={'k': 1})

    retrieval_chain =create_retrieval_chain(
        retriever,
        chain
    )
    return retrieval_chain


docs = get_document_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


response = chain.invoke({
    'input':"What is LCEL",
})

print(response['answer'])