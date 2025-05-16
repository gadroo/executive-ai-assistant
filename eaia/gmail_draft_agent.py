"""Agent responsible for creating Gmail drafts using the Gmail API and Anthropic."""

import os
import logging
from typing import List, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from eaia.gmail import get_credentials

logger = logging.getLogger(__name__)

class DraftEmailInput(BaseModel):
    """Input for drafting an email."""
    subject: str = Field(description="The subject of the email")
    to: List[str] = Field(description="List of recipient email addresses")
    content: str = Field(description="The content of the email to draft")

@tool(args_schema=DraftEmailInput)
def create_gmail_draft(subject: str, to: List[str], content: str) -> str:
    """
    Creates a draft email in Gmail without sending it.
    
    Args:
        subject: The subject line of the email
        to: List of recipients' email addresses
        content: The body of the email
    
    Returns:
        A message indicating the draft was created successfully
    """
    creds = get_credentials()
    
    from googleapiclient.discovery import build
    from email.mime.text import MIMEText
    import base64
    
    service = build('gmail', 'v1', credentials=creds)
    
    message = MIMEText(content)
    message['to'] = ', '.join(to)
    message['subject'] = subject
    
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    create_message = {
        'message': {
            'raw': encoded_message
        }
    }
    
    draft = service.users().drafts().create(userId='me', body=create_message).execute()
    return f"Draft created successfully with ID: {draft['id']}"

class SearchDraftsInput(BaseModel):
    """Input for searching drafts in Gmail."""
    query: str = Field(description="The search query to find drafts")
    max_results: int = Field(default=5, description="Maximum number of results to return")

@tool(args_schema=SearchDraftsInput)
def search_gmail_drafts(query: str, max_results: int = 5) -> str:
    """
    Searches for draft emails in Gmail.
    
    Args:
        query: The search query to find drafts
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        A formatted string with the search results
    """
    creds = get_credentials()
    
    from googleapiclient.discovery import build
    import base64
    
    service = build('gmail', 'v1', credentials=creds)
    
    # Get all drafts first
    drafts_response = service.users().drafts().list(userId='me').execute()
    drafts = drafts_response.get('drafts', [])
    
    if not drafts:
        return "No drafts found."
    
    matching_drafts = []
    for draft in drafts[:max_results]:
        draft_data = service.users().drafts().get(userId='me', id=draft['id']).execute()
        message = draft_data['message']
        
        # Get message details
        headers = message['payload']['headers']
        subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
        
        # Get message body
        if 'parts' in message['payload']:
            body_data = message['payload']['parts'][0]['body'].get('data', '')
        else:
            body_data = message['payload']['body'].get('data', '')
            
        if body_data:
            body_text = base64.urlsafe_b64decode(body_data).decode('utf-8')
        else:
            body_text = 'No content'
        
        # Check if the draft matches the query
        if query.lower() in subject.lower() or query.lower() in body_text.lower():
            matching_drafts.append({
                'id': draft['id'],
                'subject': subject,
                'snippet': body_text[:100] + '...' if len(body_text) > 100 else body_text
            })
    
    if not matching_drafts:
        return f"No drafts found matching '{query}'."
    
    result = f"Found {len(matching_drafts)} drafts matching '{query}':\n\n"
    for i, draft in enumerate(matching_drafts, 1):
        result += f"{i}. Subject: {draft['subject']}\n"
        result += f"   Snippet: {draft['snippet']}\n"
        result += f"   Draft ID: {draft['id']}\n\n"
    
    return result

def create_gmail_agent():
    """
    Creates a LangChain agent that can interact with Gmail using the GmailToolkit.
    Uses Anthropic's Claude instead of OpenAI.
    
    Returns:
        A LangChain agent configured to use the Gmail toolkit and Anthropic
    """
    # Initialize the Gmail toolkit
    toolkit = GmailToolkit()
    
    # Get all tools from the toolkit and add our custom tools
    tools = toolkit.get_tools()
    tools.extend([create_gmail_draft, search_gmail_drafts])
    
    # Initialize the Anthropic LLM
    llm = ChatAnthropic(
        temperature=0,
        model="claude-3-5-sonnet-latest"  # Use the appropriate Claude model
    )
    
    # Create and return the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

def draft_email(prompt: str):
    """
    Uses the Gmail agent to draft an email based on a prompt.
    
    Args:
        prompt: A description of the email to be drafted
    
    Returns:
        The response from the agent
    """
    agent = create_gmail_agent()
    response = agent.invoke(
        f"Create a gmail draft for the following: {prompt}. "
        f"Under no circumstances may you send the message, only create a draft."
    )
    return response

def search_drafts(query: str):
    """
    Uses the Gmail agent to search for drafts.
    
    Args:
        query: The search query to find drafts
    
    Returns:
        The response from the agent listing found drafts
    """
    agent = create_gmail_agent()
    response = agent.invoke(f"Search in my drafts for: {query}")
    return response 