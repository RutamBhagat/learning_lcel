from typing import List, Tuple

from langchain_core.agents import AgentAction


# Note: This is just for learning purposes. The actual implementation is already in langchain
# from langchain.agents.format_scratchpad import format_log_to_str
def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process"""
    thoughts = []
    for action, observation in intermediate_steps:
        thoughts.append(llm_prefix + action.log)
        thoughts.append(observation_prefix + observation)
    return "\n".join(thoughts)
