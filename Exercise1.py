# Exercise 1: Create a Prompt Template using from_messages
# After complete lesson2.py 
# Task:
# 1. Modify your prompt so that it contains a system message.
# 2. The system message should instruct the model to generate a motivational quote.
# 3. The user message should pass the topic dynamically.
# 4. Invoke the chain with "Innovation" as the topic.
# 5. Print the generated response.

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Initialize the ChatGroq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0
)

# Step 2: Create a Prompt Template using from_messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a short motivational quote about a given topic."),
    ("user", "{topic}")
])

# Step 3: Create the LLM Chain
chain = prompt | llm

# Step 4: Invoke the Chain with the topic "Innovation"
response = chain.invoke({"topic" : "Innovation"})

# Step 5: Print the Generated Response
print(response.content)
