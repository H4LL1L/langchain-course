from dotenv import load_dotenv
load_dotenv()
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent




tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")
agent= create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():

    result = chain.invoke(
        input = {
            "input": " search for 3 job postings for an ai engineer in the bay area on linkedin and list their details you should find the jobs at Turkey you have to find at turkey",}
    )
    print(result)

if __name__ == "__main__":
    main()
