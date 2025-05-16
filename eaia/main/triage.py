"""Agent responsible for triaging the email, can either ignore it, try to respond, or notify user."""

from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import RemoveMessage
from langgraph.store.base import BaseStore
from langchain_core.output_parsers import JsonOutputParser
import asyncio
import time
import re
from anthropic import InternalServerError

from eaia.schemas import (
    State,
    RespondTo,
)
from eaia.main.fewshot import get_few_shot_examples
from eaia.main.config import get_config_async

# Minimal essential patterns - just enough to catch the most common cases
# Critical set of promotional patterns that should be filtered
PROMOTIONAL_INDICATORS = [
    (r"newsletter", r"unsubscribe", 0.9),  # Newsletter with unsubscribe link
    (r"nvidia.*news", "", 0.8),            # NVIDIA news specifically mentioned
    (r"weekly digest", "", 0.7),           # Weekly digest emails
    (r"personalized.*for you", "", 0.7),   # Personalized content emails
]

# Critical career and meeting patterns that should never be ignored
IMPORTANT_PATTERNS = [
    r"assignment review",
    r"interview",
    r"job offer",
    r"meeting invitation",
    r"calendar invite",
]

def should_ignore_immediately(email):
    """
    Check if an email should be immediately ignored based on minimal pattern matching.
    This is only used for the most obvious promotional content.
    Most decisions are left to the LLM.
    """
    subject = email.get("subject", "").lower()
    content = email.get("page_content", "").lower()
    from_email = email.get("from_email", "").lower()
    
    # Always preserve important career emails
    if any(re.search(pattern, subject, re.IGNORECASE) or re.search(pattern, content, re.IGNORECASE) 
           for pattern in IMPORTANT_PATTERNS):
        return False
    
    # Meeting invitations should generally not be ignored
    if (("invitation" in subject or "invited" in subject or "calendar" in subject) and 
        any(marker in subject for marker in ["@", "at", "pm", "am"])):
        return False

    # Check for promotional content using minimal patterns
    for subject_pattern, content_pattern, confidence in PROMOTIONAL_INDICATORS:
        subject_match = re.search(subject_pattern, subject, re.IGNORECASE) if subject_pattern else False
        content_match = re.search(content_pattern, content, re.IGNORECASE) if content_pattern else False
        
        if (subject_match and content_match and confidence > 0.8) or \
           (subject_match and confidence > 0.9) or \
           (content_match and confidence > 0.9):
            print(f"Ignoring obvious promotional email: {subject}")
            return True
            
    # Let the LLM handle most of the decisions
    return False

triage_prompt = """You are {full_name}'s executive assistant. Help {name} manage emails efficiently.

{background}

YOUR ROLE:
You must determine how to handle each email by classifying it as:
- IGNORE (respond 'no'): Promotional content, newsletters, mass emails, irrelevant updates
- RESPOND (respond 'email'): Messages requiring a direct response
- NOTIFY (respond 'notify'): Important information {name} should know but doesn't need to respond to

SMART TRIAGE GUIDELINES:
- Career advancement is the highest priority (respond or notify)
- Meeting invitations for interviews, reviews, or professional development are critical
- Academic communications from professors or about coursework require responses
- Technical opportunities aligned with AI/ML interests deserve attention
- Personal communications from friends/family should receive responses
- Mass emails, newsletters, marketing, and promotional content should be ignored

IGNORE THESE (examples):
{triage_no}

RESPOND TO THESE (examples):
{triage_email}

NOTIFY ABOUT THESE (examples):
{triage_notify}

PRIORITIZE BASED ON CONTEXT:
- Sender relationship (professor > recruiter > marketing)
- Content relevance to career goals and current projects
- Action requirements (direct questions need responses)
- Time sensitivity (deadlines, scheduled meetings)
- Professional value (networking, skill development)

{fewshotexamples}

Analyze the following email and use your judgment about what would be most helpful for {name}:
From: {author}
To: {to}
Subject: {subject}
{email_thread}

Respond with JSON: {{ "logic": "your reasoning here", "response": "no" | "email" | "notify" | "question" }}"""


async def retry_with_exponential_backoff(func, max_retries=5, initial_delay=1):
    """Retry a function with exponential backoff."""
    retries = 0
    delay = initial_delay
    
    while True:
        try:
            return await func()
        except InternalServerError as e:
            # Check for overloaded error
            if "overloaded_error" in str(e).lower() and retries < max_retries:
                retries += 1
                wait_time = delay * (2 ** (retries - 1)) 
                print(f"Anthropic API overloaded. Retrying in {wait_time} seconds (attempt {retries}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                # If it's not an overload error or we've exceeded retries, re-raise
                raise


async def triage_input(state: State, config: RunnableConfig, store: BaseStore):
    # Only filter the most obvious newsletter/promotional content
    # Let the AI make most of the decisions based on context
    if should_ignore_immediately(state["email"]):
        # Create a RespondTo object with "no" as the response
        response = RespondTo(
            logic="This is clearly promotional content that can be safely ignored.",
            response="no"
        )
        
        if len(state["messages"]) > 0:
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
            return {"triage": response, "messages": delete_messages}
        else:
            return {"triage": response}
    
    # Rely on the AI for most triage decisions
    model = config["configurable"].get("model", "claude-3-5-sonnet-latest")
    llm = ChatAnthropic(model=model, temperature=0)
    examples = await get_few_shot_examples(state["email"], store, config)
    prompt_config = await get_config_async(config)
    input_message = triage_prompt.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        to=state["email"].get("to_email", ""),
        subject=state["email"]["subject"],
        fewshotexamples=examples,
        name=prompt_config["name"],
        full_name=prompt_config["full_name"],
        background=prompt_config["background"],
        triage_no=prompt_config["triage_no"],
        triage_email=prompt_config["triage_email"],
        triage_notify=prompt_config["triage_notify"],
    )
    
    # Use JsonOutputParser with RespondTo as the model
    parser = JsonOutputParser(pydantic_object=RespondTo)
    chain = llm | parser
    
    # Use retry logic for API calls that might fail due to overload
    async def call_api():
        return await chain.ainvoke(input_message)
    
    response = await retry_with_exponential_backoff(call_api)
    
    if len(state["messages"]) > 0:
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        return {"triage": response, "messages": delete_messages}
    else:
        return {"triage": response}
