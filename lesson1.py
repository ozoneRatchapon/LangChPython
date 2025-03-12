# Chat Model 
# 1. Show simple example of using the ChatGroq model
# 2. Show simple example of using the ChatOllama model
# 3. Show simple example of using the ChatGoogleGenerativeAI model
# 4. Show invoking the model with a prompt
# 5. Show Batch invoking the model with a prompt
# 6. Show Streaming invoking the model with a prompt

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
)

# localLLM = ChatOllama(
#     model="llama3.2:1b",
#     temperature=0,
# )

# googleLLM = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

response = llm.invoke("Write a poem about AI")
print(response)
print(response.content)

# response = llm.batch(["How are you","Write a poem about AI"])
# print('---- response ----')
# print(response[0].content)
# print('---- response ----')
# print(response[1].content)

# response = llm.stream("Write a poem about AI")
# for chunk in response:
#     print(chunk.content, end="", flush=True)

