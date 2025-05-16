"""Node for integrating the Gmail draft agent with LangGraph."""

from typing import Dict, List, Any, TypedDict, Optional
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage

from eaia.gmail_draft_agent import create_gmail_agent
from eaia.schemas import State
from eaia.main.config import get_config_async, get_config_sync

logger = logging.getLogger(__name__)

class GmailDraftInput(BaseModel):
    """Input for drafting an email through the Gmail API."""
    content: str = Field(description="Content for the email draft")
    subject: str = Field(description="Subject line for the email draft")
    to: List[str] = Field(description="Recipients for the email")

class GmailSearchInput(BaseModel):
    """Input for searching Gmail drafts."""
    query: str = Field(description="Query string to search for in drafts")

@tool(args_schema=GmailDraftInput)
def draft_gmail_message(content: str, subject: str, to: List[str]) -> str:
    """
    Creates a draft email in Gmail without sending it.
    
    Args:
        content: The body of the email
        subject: The subject line of the email
        to: List of recipient email addresses
        
    Returns:
        A message confirming the draft was created
    """
    agent = create_gmail_agent()
    command = (
        f"Create a Gmail draft with the following details:\n"
        f"Subject: {subject}\n"
        f"To: {', '.join(to)}\n"
        f"Content:\n{content}\n\n"
        f"Do NOT send the email, only create a draft."
    )
    response = agent.invoke(command)
    return str(response)

@tool(args_schema=GmailSearchInput)
def search_gmail_drafts(query: str) -> str:
    """
    Searches for draft emails in Gmail.
    
    Args:
        query: The search query to find drafts
        
    Returns:
        A formatted string with the search results
    """
    agent = create_gmail_agent()
    command = f"Search in my Gmail drafts for: {query}"
    response = agent.invoke(command)
    return str(response)

async def _gmail_agent_node_impl(state: State, config: Dict[str, Any], store: Any = None) -> Dict[str, Any]:
    """
    Implementation of the Gmail agent node functionality.
    
    Args:
        state: The current state of the conversation
        config: Configuration for the node
        store: The LangGraph store (optional)
        
    Returns:
        Updated state with the Gmail agent's response
    """
    model = config["configurable"].get("model", "claude-3-5-sonnet-latest")
    llm = ChatAnthropic(model=model, temperature=0)
    
    # Configure model with Gmail tools
    llm_with_tools = llm.bind_tools([draft_gmail_message, search_gmail_drafts])
    
    # Get email context from state
    email_data = state["email"]
    
    # Get config - always use the async version and properly handle errors
    try:
        # If "email" is directly in the configurable, we can use it directly without file I/O
        if "email" in config["configurable"]:
            prompt_config = config["configurable"]
        else:
            # Otherwise we need to load from the config file using the async method
            prompt_config = await get_config_async(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        # Provide a minimal default config if all else fails
        prompt_config = {
            "full_name": "Executive Assistant",
            "name": "Assistant"
        }
    
    # Construct a prompt for the LLM
    prompt = f"""You are {prompt_config.get('full_name', 'Executive Assistant')}'s executive assistant. 
    
I need you to help with Gmail drafts related to the following email:

From: {email_data.get('from_email', 'Unknown')}
To: {email_data.get('to_email', 'Unknown')}
Subject: {email_data.get('subject', 'No Subject')}

Content:
{email_data.get('page_content', 'No content')}

Based on this email, what would you like to do? You can:
1. Draft a reply email using the draft_gmail_message tool
2. Search existing drafts using the search_gmail_drafts tool

Please use the appropriate tool based on what seems most helpful.
"""
    
    # Call the LLM with tools
    messages = [HumanMessage(content=prompt)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Add the response to the messages
    return {"messages": [response]}

# Create a wrapper function that handles any parameter configuration
async def gmail_agent_node(state: State, config: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    LangGraph node that provides Gmail draft functionality within the Executive AI Assistant.
    This wrapper handles compatibility with different LangGraph versions and parameter configurations.
    
    Args:
        state: The current state of the conversation
        config: Configuration for the node
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        
    Returns:
        Updated state with the Gmail agent's response
    """
    # Extract store if it exists in args or kwargs
    store = None
    if args and len(args) > 0:
        store = args[0]
    elif 'store' in kwargs:
        store = kwargs['store']
        
    return await _gmail_agent_node_impl(state, config, store)

# Conditional function to determine next node in the graph
def should_use_gmail_agent(state: State) -> str:
    """
    Determines if the Gmail agent node should be used based on state.
    
    Args:
        state: The current conversation state
        
    Returns:
        Next node name to route to
    """
    # Check if the triage suggests an email response
    triage = state.get("triage", None)
    if triage and hasattr(triage, "response") and triage.response == "email":
        # Check if the email seems to be about drafting or managing emails
        email_content = state["email"].get("page_content", "").lower()
        email_subject = state["email"].get("subject", "").lower()
        
        keywords = ["draft", "gmail", "email drafts", "save email", "prepare email", "write email"]
        
        if any(keyword in email_content.lower() for keyword in keywords) or \
           any(keyword in email_subject.lower() for keyword in keywords):
            return "gmail_agent_node"
    
    # Default flow
    return "draft_response" 