from dotenv import load_dotenv
import sys
import asyncio
from pprint import pprint

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv(override=True)

client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=America/New_York"
            ]
        }
    }
)

# Connect to an online MCP server that provides a tool for getting the current time in a specified timezone
async def connect_online_mcp():
    tools = await client.get_tools()
    print("Tools available from local MCP server:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    #-----------------------------------------



    # -------------------------------------------
    agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
    )

    question = HumanMessage(content="What time is it?")

    response = await agent.ainvoke(
        {"messages": [question]}
    )

    pprint(response)

# uv run python  -m project_langchain.connect_online_mcp_server
if __name__ == '__main__':
    asyncio.run(connect_online_mcp())


