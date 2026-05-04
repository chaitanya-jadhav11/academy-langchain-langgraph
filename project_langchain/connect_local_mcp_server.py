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
        "local_server": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "-m", "project_langchain.resources.mcp_server"],
            }
    }
)

# connect to local MCP server that provides tools for searching the web and accessing langchain-ai repo files
async def connect_local_mcp():
    tools = await client.get_tools()
    print("Tools available from local MCP server:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    #-----------------------------------------

    # get resources
    resources = await client.get_resources("local_server")
    print("\nResources available from local MCP server:")
    for resource in resources:
        print(f"- {resource.metadata=}")
    #-----------------------------------------

    # get prompts
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content
    print("\nPrompt template from local MCP server:")
    print(prompt)


    # -------------------------------------------
    agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
        system_prompt=prompt
    )

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config
    )
    pprint(response)

# uv run python  -m project_langchain.connect_local_mcp_server
if __name__ == '__main__':
    asyncio.run(connect_local_mcp())


