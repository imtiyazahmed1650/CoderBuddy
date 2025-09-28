from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from langchain.globals import set_verbose, set_debug
from typing import List,Union
from langgraph.graph import StateGraph   , END
from langgraph.prebuilt import create_react_agent
import os

from prompts import *
from states import *
from tools import *


load_dotenv()

set_debug(True)
set_verbose(True)


api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY is not set in the environment or .env file.")


llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-120b")


def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan":resp}


def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))

    resp.plan = plan
    return {"task_plan": resp}


# Coder Agent
def coder_agent(state:dict) -> dict:
    """LangGraph tool-using coder agent."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    react_agent.invoke({"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]})

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}



graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architecture",architect_agent)
graph.add_node("coder",coder_agent)

graph.set_entry_point("planner")
graph.add_edge("planner","architecture")
graph.add_edge("architecture","coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

#graph.add_edge("coder",END)


agent = graph.compile()

user_prompt = "Create a simple web Application"

if __name__ == "__main__":
    result = agent.invoke({"user_prompt":user_prompt},{"recursion_limit": 100})
    print(result)



# try:

#     res = 
#     print('Response is ',res)
# except Exception as e:
#     print("❌ Error during LLM invocation:", e)



