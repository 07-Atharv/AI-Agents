import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# API keys (already set in environment variables)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# LLM
groq_llm = ChatGroq(model="llama3-70b-8192")

# Tool
search_tool = TavilySearchResults(max_results=2)

# System prompt
system_prompt = "Act as an AI Pro chatbot who is smart and has deep research skills."


def response_from_ai_agent(llm_id, query_text, allow_search, prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        raise ValueError("Unsupported provider.")

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # Create the agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt or system_prompt
    )

    # Messages must be a list of HumanMessage / AIMessage
    input_state = {
        "messages": [HumanMessage(content=query_text)]
    }

    # Invoke the agent
    response = agent.invoke(input_state)

    # Extract the latest AI response
    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    return ai_messages[-1] if ai_messages else "No response received."


# 🧪 Sample Call:
response = response_from_ai_agent(
    llm_id="llama3-70b-8192",
    query_text="Tell me about the crypto market.",
    allow_search=True,
    prompt=system_prompt,
    provider="Groq"
)

print("Response:", response)
