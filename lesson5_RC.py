# Retrieval Chain : load the document from website.
# 1. Create a document from website

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader


def  get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

#print(get_document_from_web("https://python.langchain.com/v0.1/docs/expression_language/"))
# docs = get_document_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
docs = get_document_from_web("https://graduate.mahidol.ac.th/inter/")


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

# chain = prompt | llm

chain = create_stuff_documents_chain(
    llm =llm,
    prompt = prompt
    )

response = chain.invoke({
    # 'input':"What is the expression language?",
    'input':"how to apply for the graduate program?",
    'context': docs
})
print(response)