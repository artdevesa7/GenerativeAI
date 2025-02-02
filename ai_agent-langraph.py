import langraph as lg
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
import re

# Define AI model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Define a web search tool
search = GoogleSearchAPIWrapper()

# Define a computation tool
@tool
def simple_calculator(expression: str) -> str:
    """Evaluates basic mathematical expressions."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize agent with available tools
tools = [Tool.from_function(name="WebSearch", func=search.run, description="Fetch web results"),
         simple_calculator]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Define AI Agent Workflow in Langraph
graph = lg.Graph()

# Define states
@graph.state()
def process_input(user_query: str):
    """Determines what action to take based on the user query."""
    if re.search(r'\b(?:calculate|solve|compute)\b', user_query, re.IGNORECASE):
        return "computation", user_query
    elif re.search(r'\b(?:search|lookup|find)\b', user_query, re.IGNORECASE):
        return "web_search", user_query
    else:
        return "ai_response", user_query

@graph.state()
def computation(expression: str):
    """Performs basic mathematical calculations."""
    result = simple_calculator.run(expression)
    return "response", result

@graph.state()
def web_search(query: str):
    """Fetches results from the web for the given query."""
    result = search.run(query)
    return "response", result

@graph.state()
def ai_response(query: str):
    """Processes general AI queries."""
    response = llm.predict(query)
    return "response", response

@graph.state()
def response(output: str):
    """Returns the final response to the user."""
    print(f"AI Response: {output}")

# Define transitions between states
graph.add_edge("process_input", "computation", condition=lambda state: state[0] == "computation")
graph.add_edge("process_input", "web_search", condition=lambda state: state[0] == "web_search")
graph.add_edge("process_input", "ai_response", condition=lambda state: state[0] == "ai_response")
graph.add_edge("computation", "response")
graph.add_edge("web_search", "response")
graph.add_edge("ai_response", "response")

# Compile the agent
agent_executor = graph.compile()

# Run example queries
if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    agent_executor.invoke("process_input", user_query=user_input)
