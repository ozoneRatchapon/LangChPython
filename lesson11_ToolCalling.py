from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# response = llm.invoke("Write a poem about AI")
# print(response)

#==========================================================

from langchain_core.tools import tool
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)
#==========================================================
from langchain_core.messages import HumanMessage  
query = "What is 44 * 12? and what is 5 + 2"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
# print(ai_msg)
messages.append(ai_msg) # Add the AI response to the messages list
# print(ai_msg.tool_calls)

#==========================================================
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

#==========================================================
print(llm_with_tools.invoke(messages).content)
