# LangChPython Workspace

This workspace contains various lessons and exercises focused on using LangChain for building language models and document retrieval systems. Below is an overview of the files and their purposes.

## File Structure

```
.venv
.env
.gitignore
cspell.json
Exercise1.py
Exercise2.py
Exercise3.py
Exercise4.py
Exercise5.py
Exercise6.py
lesson1.py
lesson2.py
lesson3.py
lesson4_RC.py
lesson5_RC.py
lesson6_RC.py
lesson7_RC.py
lesson8_chatbot.py
lesson9_chatbot.py
lesson10_chatbot.py
lesson11_ToolCalling.py
lesson12_RC_1.py
requirements.txt
```

## Files and Descriptions

### Lessons

- **lesson1.py**: Introduces the ChatGroq model and demonstrates simple examples of using different chat models.
- **lesson2.py**: Covers prompt templates and LLM chains, including creating prompt templates from templates and messages.
- **lesson3.py**: Explores output parsers, including string, list, and JSON output parsers.
- **lesson4_RC.py**: Demonstrates creating and invoking a document retrieval chain.
- **lesson5_RC.py**: Shows how to load documents from a website and create a document retrieval chain.
- **lesson6_RC.py**: Loads documents from a website, splits them, and stores them in a vector store.
- **lesson7_RC.py**: Retrieves answers from a vector store after loading and splitting documents.
- **lesson8_chatbot.py**: Builds a chatbot that connects to the web and answers questions via chat history.
- **lesson9_chatbot.py**: Similar to lesson8, but includes creating and processing chat history.
- **lesson10_chatbot.py**: Adds a retrieval chain to the chatbot.
- **lesson11_ToolCalling.py**: Demonstrates tool calling with the ChatGroq model.
- **lesson12_RC_1.py**: Loads documents from a website, splits them, and stores them in a Pinecone vector store.

### Exercises

- **Exercise1.py**: Creates a prompt template using `from_messages` and generates a motivational quote.
- **Exercise2.py**: Generates a list of planets using `CommaSeparatedListOutputParser`.
- **Exercise3.py**: Generates tips on a given topic without using an output parser.
- **Exercise4.py**: Uses `StrOutputParser` to ensure the output is always a string.
- **Exercise5.py**: Loads and splits documents from the web.
- **Exercise6.py**: Builds a document retrieval system that fetches, splits, embeds, and stores documents in a vector store.

### Configuration Files

- **.env**: Environment variables.
- **.gitignore**: Specifies files to be ignored by Git.
- **cspell.json**: Configuration for the spell checker.
- **requirements.txt**: Lists the dependencies required for the project.

### Virtual Environment

- **.venv**: Contains the virtual environment for the project, including the Python interpreter and all installed dependencies. This directory is typically excluded from version control.

## Getting Started

1. **Install Dependencies**: Ensure you have all the required dependencies installed by running:
    ```sh
    pip install -r requirements.txt
    ```

2. **Set Up Environment Variables**: Create a `.env` file with the necessary environment variables.

3. **Run Lessons and Exercises**: Execute the Python files to see the examples and exercises in action. For example:
    ```sh
    python lesson1.py
    ```

## Key Concepts

- **LangChain**: A framework for building language models and document retrieval systems.
- **ChatGroq**: A language model used for generating responses based on prompts.
- **Prompt Templates**: Templates used to structure the input to the language model.
- **Output Parsers**: Parsers used to format the output from the language model.
- **Vector Stores**: Databases used to store and retrieve document embeddings.

## Example Usage

### Loading and Splitting Documents

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_document(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return split_docs

url = "https://python.langchain.com/v0.1/docs/expression_language/"
split_documents = load_and_split_document(url)
print(f"Total number of chunks: {len(split_documents)}")
```

### Creating a Retrieval Chain

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_together import TogetherEmbeddings

def create_chain(vectorStore):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template("Answer the user's question: Context :  {context} User Question : {input}")
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vectorStore.as_retriever(search_kwargs={'k': 3})
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain
```

## Conclusion

This workspace provides a comprehensive set of lessons and exercises to help you understand and implement various aspects of LangChain, including prompt templates, output parsers, and document retrieval systems. Explore the files and run the examples to deepen your understanding of these concepts.