import asyncio
from typing import Any
from tavily import TavilyClient 
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
    RunConfig,
)
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

set_tracing_disabled(True)
load_dotenv(override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Tavily client
tavily_client = TavilyClient(api_key=tavily_api_key)

# Wrap Tavily search as a function_tool
@function_tool
def web_search(query: str) -> str:
    """Search the web using Tavily API and return summarized results."""
    results = tavily_client.search(query)
    return "\n".join([res["content"] for res in results["results"]])

# Gemini external client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

web_agent = Agent(
    name="websearch_tool_agent",
    instructions="You are a helpful AI agent that can search and extract information using Tavily.",
    tools=[web_search],  
    model=model
)

async def main():
    msg = input("Enter your query : ")
    result = await Runner.run(web_agent, msg)
    print(result.final_output)

asyncio.run(main())
