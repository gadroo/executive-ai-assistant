"""Core agent responsible for drafting emails for a final-year college student."""

from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langgraph.store.base import BaseStore
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import json
from anthropic import InternalServerError
from langchain_core.exceptions import OutputParserException

from eaia.schemas import (
    State,
    NewEmailDraft,
    ResponseEmailDraft,
    Question,
    MeetingAssistant,
    SendCalendarInvite,
    Ignore,
    email_template,
)
from eaia.main.config import get_config_async

EMAIL_WRITING_INSTRUCTIONS = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

{background}

{name} receives numerous emails daily. This email has been determined to require {name}'s response.

Your primary responsibility is to draft responses that authentically represent {name}'s voice while maintaining perfect contextual awareness of the relationship with the sender.

# Personal Email Handling

For personal emails (like check-ins from friends, family, or social acquaintances):
- ALWAYS generate an automatic response using the ResponseEmailDraft tool
- Never use the Question tool for personal emails
- Draft warm, friendly responses that maintain {name}'s authentic voice
- Include appropriate pleasantries and reciprocal questions
- For casual meetup requests, suggest potential times while keeping {name}'s schedule in mind
- Ensure responses sound natural and personal, not automated

# Using the `Question` tool

For non-personal emails, ensure you have complete information before drafting. If critical details are missing, use the Question tool to ask {name} directly.

NEVER include placeholders or assumptions in your drafts. Obtain precise information regarding:
- Specific details about job opportunities or internships
- Event availability and scheduling preferences
- {name}'s interest level in proposed collaborations
- Current academic commitments and deadlines
- Relationship context with the sender
- Technical project specifications

If someone requests {name}'s attendance, participation, or commitment to any event, meeting, or project, do not confirm unless {name} has explicitly approved. Always verify availability and interest before committing.

When uncertain about {name}'s preferences or missing crucial information, use the Question tool rather than making assumptions. For scheduling inquiries, defer to the MeetingAssistant rather than asking {name} directly about availability.

# Using the `ResponseEmailDraft` tool

Once you have sufficient information, draft responses using the ResponseEmailDraft tool.

Always write as if you are {name} directly. Never identify yourself as an assistant or third party.

For recipient management:
- Only add new recipients when explicitly instructed by {name}
- Verify you have the correct email addresses before adding
- Do not duplicate recipients already included in the thread
- Never fabricate email addresses

Adapt your communication style based on relationship context:
- Personal communications: Warm, friendly, and conversational while maintaining {name}'s authentic voice
- Academic communications: Precise, evidence-based language with appropriate citations and scholarly terminology
- Professional outreach: Achievement-focused with quantifiable results and specific technical competencies
- Networking exchanges: Value-oriented, purposeful with clear mutual benefit articulation
- Administrative correspondence: Procedural clarity with solution-oriented approaches
- Collaborative discussions: Reliable commitments with accountable timelines
- Social communications: Authentically casual while maintaining professional boundaries

{response_preferences}

# Using the `SendCalendarInvite` tool

When scheduling is required, use the SendCalendarInvite tool with confidence when:
- {name} has explicitly approved the meeting
- You've confirmed calendar availability
- The meeting aligns with {name}'s prioritization framework

Schedule according to these priorities:
1. Job interviews and career advancement opportunities
2. Academic requirements and educational commitments
3. Strategic networking with industry professionals
4. Project coordination and collaboration sessions
5. Academic advising and mentorship
6. Organizational leadership commitments
7. Social and community engagements

{schedule_preferences}

# Using the `NewEmailDraft` tool

For initiating new communication threads, use the NewEmailDraft tool when you have all essential information.

Structure new emails with:
- Clear, specific subject lines
- Appropriate greeting based on relationship context
- Concise purpose statement in the opening paragraph
- Specific, actionable requests
- Relevant context or background information
- Professional closing aligned with relationship context

When facilitating introductions that {name} has approved, draft connection emails that clearly establish value for both parties.

# Using the `MeetingAssistant` tool

For legitimate meeting scheduling requests, engage the specialized MeetingAssistant tool rather than consulting {name} directly about availability.

Only invoke this tool when:
- The request comes from a legitimate sender
- The email explicitly requests scheduling a new meeting
- Time coordination is required

For already scheduled meetings where details are being referenced or confirmed, there is no need to invoke this tool.

# Context: Critical background for authentic communication

{name}'s current priorities (in order of importance):
1. Securing post-graduation employment in AI development
2. Maintaining academic excellence and research contributions
3. Expanding professional network in the AI/ML community
4. Fulfilling campus leadership responsibilities
5. Completing graduation requirements efficiently

In all communications, project:
- Technical competence in AI and software engineering
- Professional reliability and follow-through
- Genuine engagement with ideas and opportunities
- Exceptional organizational capacity
- Authentic personality that balances technical expertise with interpersonal warmth

{random_preferences}"""

draft_prompt = """{instructions}

Remember to call tools with precise syntax and complete parameters. Use the exact tool names as specified below:

1. For Question: {{"name": "Question", "arguments": {{"content": "your question here"}}}}
2. For ResponseEmailDraft: {{"name": "ResponseEmailDraft", "arguments": {{"content": "email content here", "new_recipients": []}}}}
3. For NewEmailDraft: {{"name": "NewEmailDraft", "arguments": {{"content": "email content here", "recipients": []}}}}
4. For MeetingAssistant: {{"name": "MeetingAssistant", "arguments": {{"call": true}}}}
5. For SendCalendarInvite: {{"name": "SendCalendarInvite", "arguments": {{"emails": [], "title": "Meeting Title", "start_time": "2024-07-01T14:00:00", "end_time": "2024-07-01T15:00:00"}}}}
6. For Ignore: {{"name": "Ignore", "arguments": {{"ignore": true}}}}

Here is the complete email thread. Analyze the entire conversation with special attention to the most recent message.

{email}"""


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
        except Exception as e: # Catch other potential exceptions during the LLM call
            if retries < max_retries:
                retries += 1
                wait_time = delay * (2 ** (retries - 1))
                print(f"LLM call failed. Retrying in {wait_time} seconds (attempt {retries}/{max_retries}). Error: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise # Re-raise after max retries


async def draft_response(state: State, config: RunnableConfig, store: BaseStore):
    """Write an email response for a popular final-year student balancing job applications, academics, and social life."""
    model = config["configurable"].get("model", "claude-3-5-sonnet-latest")
    
    tools = [
        NewEmailDraft,
        ResponseEmailDraft,
        Question,
        MeetingAssistant,
        SendCalendarInvite,
    ]
    messages = state.get("messages") or []
    if len(messages) > 0:
        tools.append(Ignore)
    
    # Bind tools to the LLM for structured output
    llm = ChatAnthropic(
        model=model,
        temperature=0,
    ).bind_tools(tools)
    
    # Use the existing async function
    prompt_config = await get_config_async(config)
    namespace = (config["configurable"].get("assistant_id", "default"),)
    key = "schedule_preferences"
    if hasattr(store, "aget"):
        result = await store.aget(namespace, key)
    else:
        result = await asyncio.to_thread(store.get, namespace, key)
    if result and "data" in result.value:
        schedule_preferences = result.value["data"]
    else:
        if hasattr(store, "aput"):
            await store.aput(namespace, key, {"data": prompt_config["schedule_preferences"]})
        else:
            await asyncio.to_thread(store.put, namespace, key, {"data": prompt_config["schedule_preferences"]})
        schedule_preferences = prompt_config["schedule_preferences"]
    key = "random_preferences"
    if hasattr(store, "aget"):
        result = await store.aget(namespace, key)
    else:
        result = await asyncio.to_thread(store.get, namespace, key)
    if result and "data" in result.value:
        random_preferences = result.value["data"]
    else:
        if hasattr(store, "aput"):
            await store.aput(namespace, key, {"data": prompt_config["background_preferences"]})
        else:
            await asyncio.to_thread(store.put, namespace, key, {"data": prompt_config["background_preferences"]})
        random_preferences = prompt_config["background_preferences"]
    key = "response_preferences"
    if hasattr(store, "aget"):
        result = await store.aget(namespace, key)
    else:
        result = await asyncio.to_thread(store.get, namespace, key)
    if result and "data" in result.value:
        response_preferences = result.value["data"]
    else:
        if hasattr(store, "aput"):
            await store.aput(namespace, key, {"data": prompt_config["response_preferences"]})
        else:
            await asyncio.to_thread(store.put, namespace, key, {"data": prompt_config["response_preferences"]})
        response_preferences = prompt_config["response_preferences"]
    _prompt = EMAIL_WRITING_INSTRUCTIONS.format(
        schedule_preferences=schedule_preferences,
        random_preferences=random_preferences,
        response_preferences=response_preferences,
        name=prompt_config["name"],
        full_name=prompt_config["full_name"],
        background=prompt_config["background"],
    )
    input_message = draft_prompt.format(
        instructions=_prompt,
        email=email_template.format(
            email_thread=state["email"]["page_content"],
            author=state["email"]["from_email"],
            subject=state["email"]["subject"],
            to=state["email"].get("to_email", ""),
        ),
    )

    # Use retry logic primarily for API errors/transient issues
    async def invoke_llm_with_retry():
        messages = [HumanMessage(content=input_message)]
        # Use the async version directly since this should already be async
        response = await llm.ainvoke(messages)
        # Ensure the response has tool calls if expected
        if not response.tool_calls:
            # For personal emails, automatically generate a response
            if is_personal_email(state["email"]):
                # Create a default ResponseEmailDraft tool call
                print("Creating default ResponseEmailDraft for personal email")
                content = generate_personal_response(state["email"])
                return AIMessage(content="", tool_calls=[{
                    "name": "ResponseEmailDraft", 
                    "arguments": {"content": content, "new_recipients": []}, 
                    "id": "response_draft_tool_call"
                }])
            # Otherwise raise an error for other email types
            print("LLM did not return a tool call for non-personal email")
            raise ValueError("LLM did not return a tool call.")
        return response

    # Function to check if an email is personal based on content
    def is_personal_email(email):
        """
        Use heuristics to determine if an email is personal in nature.
        This is a backup method that only runs if the LLM fails to generate a proper tool call.
        """
        subject = email.get("subject", "").lower()
        content = email.get("page_content", "").lower()
        from_email = email.get("from_email", "").lower()
        
        # Email domains that typically send personal emails
        personal_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
        
        # Check if it's from a personal domain
        is_from_personal_domain = any(domain in from_email for domain in personal_domains)
        
        # Simple content-based check for personal language
        personal_indicators = ["catch up", "how are you", "miss you", "coffee", "lunch", "friend"]
        has_personal_language = any(indicator in content or indicator in subject for indicator in personal_indicators)
        
        # Consider short emails that start with "hi" as potentially personal
        is_short_casual = len(content.split()) < 30 and any(greeting in content[:10] for greeting in ["hi", "hey", "hello"])
        
        # Use a combination of factors to determine if it's personal
        return (is_from_personal_domain and (has_personal_language or is_short_casual)) or \
               (has_personal_language and is_short_casual)
        
    # Function to generate a personal response
    def generate_personal_response(email):
        sender_name = email.get("from_email", "").split('@')[0].split('.')
        if len(sender_name) > 0:
            sender_name = sender_name[0].capitalize()
        else:
            sender_name = "there"
            
        subject = email.get("subject", "").lower()
        content = email.get("page_content", "").lower()
        
        # Check if it's a meet-up request
        if any(phrase in content for phrase in ["meet", "catch up", "coffee", "lunch", "dinner"]):
            return f"""Hi {sender_name},

Great to hear from you! I'd love to catch up. How about sometime next week? I'm free on Tuesday afternoon or Thursday after 5.

Let me know what works for you!

Best,
{prompt_config["name"]}"""
        # Generic friendly response
        else:
            return f"""Hi {sender_name},

Great to hear from you! Thanks for reaching out. Things have been busy on my end with finishing up my degree and working on some exciting AI projects.

How have you been? What's new in your world?

Best,
{prompt_config["name"]}"""

    try:
        # Use retry_with_exponential_backoff which now handles more exceptions during the call
        result = await retry_with_exponential_backoff(invoke_llm_with_retry)
    except Exception as e:
        # Log the exception for debugging
        print(f"Error generating response: {e}")
        
        # If we still can't get a valid tool call, create a default one for personal emails
        if is_personal_email(state["email"]):
            content = generate_personal_response(state["email"])
            result = AIMessage(content="", tool_calls=[{
                "name": "ResponseEmailDraft", 
                "arguments": {"content": content, "new_recipients": []}, 
                "id": "response_draft_fallback"
            }])
        else:
            # For non-personal emails, raise the error
            raise ValueError(f"Failed to get valid tool call from LLM after retries. Last error: {e}") from e

    return {"draft": result, "messages": [result]}