# Chatbot: Connect with web and answer question as chatbot via chat history.  
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

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder



def  get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user question base on the context: {context}"),
          MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}"),
    ])


    chain = create_stuff_documents_chain(
        llm =llm,
        prompt = prompt
        )
    
    retriever = vectorStore.as_retriever(search_kwargs={'k': 3})

    retrieval_chain =create_retrieval_chain(
        retriever,
        chain
    )
    return retrieval_chain

def process_chat(chain, input_text, chat_history):
    response = chain.invoke({
        'chat_history':chat_history,
        'input':input_text,
    })
    return response['answer']


if __name__ == "__main__":
    docs = get_document_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = [
        HumanMessage(content="hello"),
        AIMessage(content="Hi, how can I help you?"),
        HumanMessage(content="My name is Ozone")
    ]

    while True:
        user_input = input("You:")
        if user_input == "exit":
            break
        response = process_chat(chain, user_input, chat_history)
        print(response)


