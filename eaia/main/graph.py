"""Overall agent."""
import json
from typing import TypedDict, Literal, Dict, Any
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from eaia.main.triage import (
    triage_input,
    should_ignore_immediately,
)
from eaia.main.draft_response import draft_response
from eaia.main.find_meeting_time import find_meeting_time
from eaia.main.rewrite import rewrite
from eaia.main.config import get_config_sync
from langchain_core.messages import ToolMessage
from eaia.main.human_inbox import (
    send_message,
    send_email_draft,
    notify,
    send_cal_invite,
)
from eaia.gmail import (
    send_email,
    mark_as_read,
    send_calendar_invite,
)
from eaia.schemas import (
    State,
    RespondTo,
)
from eaia.main.gmail_agent_node import gmail_agent_node, should_use_gmail_agent


def initialize_state_node(state: State) -> Dict[str, Any]:
    """Initialization node that runs first to ensure all required fields are present."""
    result = {}
    
    # Always include the email from the incoming state
    if "email" in state:
        result["email"] = state["email"]
    else:
        # This is a critical error - email is required
        raise ValueError("Email must be provided in the initial state")
    
    # Initialize messages if not present
    if "messages" not in state or not state["messages"]:
        result["messages"] = []
    else:
        result["messages"] = state["messages"]
    
    # Initialize triage with a default value if not present
    if "triage" not in state:
        # Create a default RespondTo that marks as read
        result["triage"] = RespondTo(
            logic="Default triage decision for initialization",
            response="no"
        )
    else:
        result["triage"] = state["triage"]
    
    return result


def route_after_triage(
    state: State,
) -> Literal["draft_response", "gmail_agent_node", "mark_as_read_node", "notify"]:
    # First, do a minimal check for obviously promotional content
    if should_ignore_immediately(state["email"]):
        print(f"Confirmed automated filtering: {state['email'].get('subject', '')}")
        return "mark_as_read_node"
    
    # Check if this is a meeting invitation - these need special handling
    subject = state["email"].get("subject", "").lower()
    
    # Trust the AI's decision in most cases
    if state["triage"].response == "email":
        # Check if we should route to the Gmail agent
        return should_use_gmail_agent(state)
    elif state["triage"].response == "no":
        return "mark_as_read_node"
    elif state["triage"].response == "notify":
        # Simple safety check for meeting invitations
        if "invitation" in subject and any(time_marker in subject for time_marker in ["@", "at", "pm", "am"]):
            # Check if it seems career-related
            if any(career_term in subject for career_term in ["assignment", "interview", "review", "job"]):
                print(f"Important meeting detected, upgrading notification to response: {subject}")
                return "draft_response"
        return "notify"
    elif state["triage"].response == "question":
        return "draft_response"
    else:
        # Default to mark_as_read_node for any unknown triage response
        return "mark_as_read_node"


def take_action(
    state: State,
) -> Literal[
    "send_message",
    "rewrite",
    "mark_as_read_node",
    "find_meeting_time",
    "send_cal_invite",
    "bad_tool_name",
]:
    prediction = state["messages"][-1]
    
    # Check if there are any tool calls in the prediction
    if not hasattr(prediction, 'tool_calls') or not prediction.tool_calls:
        # Default to mark as read if no tool calls are present
        return "mark_as_read_node"
    
    # Take the first tool call if there are multiple
    tool_call = prediction.tool_calls[0]
    
    if tool_call["name"] == "Question":
        return "send_message"
    elif tool_call["name"] == "ResponseEmailDraft":
        return "rewrite"
    elif tool_call["name"] == "Ignore":
        return "mark_as_read_node"
    elif tool_call["name"] == "MeetingAssistant":
        return "find_meeting_time"
    elif tool_call["name"] == "SendCalendarInvite":
        return "send_cal_invite"
    else:
        return "bad_tool_name"


def bad_tool_name(state: State):
    tool_call = state["messages"][-1].tool_calls[0]
    message = f"Could not find tool with name `{tool_call['name']}`. Make sure you are calling one of the allowed tools!"
    last_message = state["messages"][-1]
    last_message.tool_calls[0]["name"] = last_message.tool_calls[0]["name"].replace(
        ":", ""
    )
    return {
        "messages": [
            last_message,
            ToolMessage(content=message, tool_call_id=tool_call["id"]),
        ]
    }


def enter_after_human(
    state,
) -> Literal[
    "mark_as_read_node", "draft_response", "send_email_node", "send_cal_invite_node"
]:
    messages = state.get("messages") or []
    if len(messages) == 0:
        if state["triage"].response == "notify":
            return "mark_as_read_node"
        # If no messages and not notify, go to draft_response
        return "draft_response"
    else:
        if isinstance(messages[-1], (ToolMessage, HumanMessage)):
            return "draft_response"
        else:
            # Check if the message has tool_calls
            if not hasattr(messages[-1], 'tool_calls') or not messages[-1].tool_calls:
                return "draft_response"
                
            execute = messages[-1].tool_calls[0]
            if execute["name"] == "ResponseEmailDraft":
                return "send_email_node"
            elif execute["name"] == "SendCalendarInvite":
                return "send_cal_invite_node"
            elif execute["name"] == "Ignore":
                return "mark_as_read_node"
            elif execute["name"] == "Question":
                return "draft_response"
            else:
                # Default to draft_response for any unknown tool
                return "draft_response"


def send_cal_invite_node(state, config):
    tool_call = state["messages"][-1].tool_calls[0]
    _args = tool_call["args"]
    email = get_config_sync(config)["email"]
    try:
        send_calendar_invite(
            _args["emails"],
            _args["title"],
            _args["start_time"],
            _args["end_time"],
            email,
        )
        message = "Sent calendar invite!"
    except Exception as e:
        message = f"Got the following error when sending a calendar invite: {e}"
    return {"messages": [ToolMessage(content=message, tool_call_id=tool_call["id"])]}


def send_email_node(state, config):
    tool_call = state["messages"][-1].tool_calls[0]
    _args = tool_call["args"]
    email = get_config_sync(config)["email"]
    new_receipients = _args["new_recipients"]
    if isinstance(new_receipients, str):
        new_receipients = json.loads(new_receipients)
    send_email(
        state["email"]["id"],
        _args["content"],
        email,
        addn_receipients=new_receipients,
    )
    return {"email": state["email"], "messages": state["messages"]}  # Ensure we return something


def mark_as_read_node(state):
    mark_as_read(state["email"]["id"])
    # Return the unchanged state to avoid invalid update errors
    return {"email": state["email"]}


def human_node(state: State):
    # Return unchanged state to prevent invalid update errors
    return {"email": state["email"], "messages": state.get("messages", [])}


class ConfigSchema(TypedDict):
    db_id: int
    model: str


graph_builder = StateGraph(State, config_schema=ConfigSchema)
graph_builder.add_node("initialize", initialize_state_node)  # Add initialization node
graph_builder.add_node(human_node)
graph_builder.add_node(triage_input)
graph_builder.add_node(draft_response)
graph_builder.add_node("gmail_agent_node", gmail_agent_node)  # Explicitly register with name and function
graph_builder.add_node(send_message)
graph_builder.add_node(rewrite)
graph_builder.add_node(mark_as_read_node)
graph_builder.add_node(send_email_draft)
graph_builder.add_node(send_email_node)
graph_builder.add_node(bad_tool_name)
graph_builder.add_node(notify)
graph_builder.add_node(send_cal_invite_node)
graph_builder.add_node(send_cal_invite)

# Set the initialize node as the entry point
graph_builder.set_entry_point("initialize")

# Add edge from initialize to triage
graph_builder.add_edge("initialize", "triage_input")

graph_builder.add_conditional_edges("triage_input", route_after_triage)
graph_builder.add_conditional_edges("draft_response", take_action)
graph_builder.add_edge("gmail_agent_node", "human_node")
graph_builder.add_edge("send_message", "human_node")
graph_builder.add_edge("send_cal_invite", "human_node")
graph_builder.add_node(find_meeting_time)
graph_builder.add_edge("find_meeting_time", "human_node")
graph_builder.add_edge("bad_tool_name", "draft_response")
graph_builder.add_edge("send_cal_invite_node", "human_node")
graph_builder.add_edge("send_email_node", "mark_as_read_node")
graph_builder.add_edge("rewrite", "send_email_draft")
graph_builder.add_edge("send_email_draft", "human_node")
graph_builder.add_edge("mark_as_read_node", END)
graph_builder.add_edge("notify", "human_node")
graph_builder.add_conditional_edges("human_node", enter_after_human)

graph = graph_builder.compile()
