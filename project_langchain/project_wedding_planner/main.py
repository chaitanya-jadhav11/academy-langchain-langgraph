import asyncio

from langchain.messages import HumanMessage
from langchain.agents import create_agent
from project_langchain.project_wedding_planner.agents import search_flights, search_venues, suggest_playlist, update_state
from project_langchain.project_wedding_planner.state import WeddingState
from dotenv import load_dotenv

load_dotenv(override=True)

# Types of orchestration
#
# 1) Prompt-driven orchestration
# 2) Router / Intent-based orchestration
# 3) Planner–Executor architecture
# 4) State machine / workflow orchestration
# 5) Graph-based orchestration (modern approach) current best practice in LangChain ecosystem
# 6) Multi-agent orchestration
# 7) Event-driven orchestration
# 8) Hybrid orchestration (what real systems use)
#
#
# Quick comparison table
# Approach	    	Control		Flexibility	 Production-readiness
# Prompt-driven		Low			High		 ❌
# Router	       	Medium		Low		     ✅
# Planner–Executor	High		Medium		 ✅✅
# State machine	    Very-High	Low			 ✅✅✅
# Graph				Very-High	High		 ✅✅✅
# Multi-agent		Medium	    High	     ⚠️
# Event-driven	 	Very High	Medium		 ✅✅✅



# Prompt-driven orchestration
coordinator = create_agent(
    model="gpt-5-nano",
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="""
    You are a wedding coordinator. 
    First find all the information you need to update the state. When you have the information, update the state.
    Once that has completed and returned, you can delegate the tasks 
    to your specialists for flights, venues, and playlists.
    Once you have received their answers, coordinate the perfect wedding for me.
    """
)


async def main():
    response = await coordinator.ainvoke(
        {
            "messages": [
                HumanMessage(content="I'm from Mumbai and I'd like a wedding in Pune for 100 guests, jazz-genre")],
        },
        config={"tags": ["WP"], "recursion_limit": 40},
        # tag traces to make them easy to find in Langsmith. Increase number of steps the agent can take to 40.
    )

    #pprint(response)
    print(response["messages"][-1].content)


# uv run python  -m project_langchain.project_wedding_planner.main
if __name__ == '__main__':
    asyncio.run(main())
