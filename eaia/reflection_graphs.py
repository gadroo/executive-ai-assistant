from langgraph.store.base import BaseStore
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command, Send

TONE_INSTRUCTIONS = "Only update the prompt to include instructions on the **style and tone and format** of the response. Do NOT update the prompt to include anything about the actual content - only the style and tone and format. The user sometimes responds differently to different types of people - take that into account, but don't be too specific."
RESPONSE_INSTRUCTIONS = "Only update the prompt to include instructions on the **content** of the response. Do NOT update the prompt to include anything about the tone or style or format of the response."
SCHEDULE_INSTRUCTIONS = "Only update the prompt to include instructions on how to send calendar invites - eg when to send them, what title should be, length, time of day, etc"
BACKGROUND_INSTRUCTIONS = "Only update the propmpt to include pieces of information that are relevant to being the user's assistant. Do not update the instructions to include anything about the tone of emails sent, when to send calendar invites. Examples of good things to include are (but are not limited to): people's emails, addresses, etc."


def get_trajectory_clean(messages):
    response = []
    for m in messages:
        response.append(m.pretty_repr())
    return "\n".join(response)


class ReflectionState(MessagesState):
    feedback: Optional[str]
    prompt_key: str
    assistant_key: str
    instructions: str


class GeneralResponse(TypedDict):
    logic: str
    update_prompt: bool
    new_prompt: str


general_reflection_prompt = """Help improve an AI agent by updating its system prompt.

Current prompt:
<current_prompt>
{current_prompt}
</current_prompt>

Agent trajectory:
<trajectory>
{trajectory}
</trajectory>

User feedback:
<feedback>
{feedback}
</feedback>

Instructions:
<instructions>
{instructions}
</instructions>

Return the full updated prompt. Include anything from before that should be kept. You can change or remove irrelevant parts. If no updates needed, return `update_prompt = False` and an empty string for new prompt."""


async def update_general(state: ReflectionState, config, store: BaseStore):
    reflection_model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    namespace = (state["assistant_key"],)
    key = state["prompt_key"]
    result = await store.aget(namespace, key)

    async def get_output(messages, current_prompt, feedback, instructions):
        trajectory = get_trajectory_clean(messages)
        prompt = general_reflection_prompt.format(
            current_prompt=current_prompt,
            trajectory=trajectory,
            feedback=feedback,
            instructions=instructions,
        )
        _output = await reflection_model.with_structured_output(
            GeneralResponse, method="json_schema"
        ).ainvoke(prompt)
        return _output

    output = await get_output(
        state["messages"],
        result.value["data"],
        state["feedback"],
        state["instructions"],
    )
    if output["update_prompt"]:
        await store.aput(
            namespace, key, {"data": output["new_prompt"]}, index=False
        )



general_reflection_graph = StateGraph(ReflectionState)
general_reflection_graph.add_node(update_general)
general_reflection_graph.add_edge(START, "update_general")
general_reflection_graph.add_edge("update_general", END)
general_reflection_graph = general_reflection_graph.compile()

MEMORY_TO_UPDATE = {
    "tone": "Instruction about the tone and style and format of the resulting email. Update this if you learn new information about the tone in which the user likes to respond that may be relevant in future emails.",
    "background": "Background information about the user. Update this if you learn new information about the user that may be relevant in future emails",
    "email": "Instructions about the type of content to be included in email. Update this if you learn new information about how the user likes to respond to emails (not the tone, and not information about the user, but specifically about how or when they like to respond to emails) that may be relevant in the future.",
    "calendar": "Instructions about how to send calendar invites (including title, length, time, etc). Update this if you learn new information about how the user likes to schedule events that may be relevant in future emails.",
}
MEMORY_TO_UPDATE_KEYS = {
    "tone": "rewrite_instructions",
    "background": "random_preferences",
    "email": "response_preferences",
    "calendar": "schedule_preferences",
}
MEMORY_TO_UPDATE_INSTRUCTIONS = {
    "tone": TONE_INSTRUCTIONS,
    "background": BACKGROUND_INSTRUCTIONS,
    "email": RESPONSE_INSTRUCTIONS,
    "calendar": SCHEDULE_INSTRUCTIONS,
}

CHOOSE_MEMORY_PROMPT = """Choose which prompts to update based on user feedback.

Agent trajectory:
<trajectory>
{trajectory}
</trajectory>

User feedback:
<feedback>
{feedback}
</feedback>

Available prompt types:
<types_of_prompts>
{types_of_prompts}
</types_of_prompts>

Select prompt types worth updating. Only choose if feedback contains relevant information. Leave empty if no updates needed."""


class MultiMemoryInput(MessagesState):
    prompt_types: list[str]
    feedback: str
    assistant_key: str


async def determine_what_to_update(state: MultiMemoryInput):
    reflection_model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    trajectory = get_trajectory_clean(state["messages"])
    types_of_prompts = "\n".join(
        [f"`{p_type}`: {MEMORY_TO_UPDATE[p_type]}" for p_type in state["prompt_types"]]
    )
    prompt = CHOOSE_MEMORY_PROMPT.format(
        trajectory=trajectory,
        feedback=state["feedback"],
        types_of_prompts=types_of_prompts,
    )

    class MemoryToUpdate(TypedDict):
        memory_types_to_update: list[str]

    response = reflection_model.with_structured_output(MemoryToUpdate).invoke(prompt)
    sends = []
    for t in response["memory_types_to_update"]:
        _state = {
            "messages": state["messages"],
            "feedback": state["feedback"],
            "prompt_key": MEMORY_TO_UPDATE_KEYS[t],
            "assistant_key": state["assistant_key"],
            "instructions": MEMORY_TO_UPDATE_INSTRUCTIONS[t],
        }
        send = Send("reflection", _state)
        sends.append(send)
    return Command(goto=sends)


# Done so this can run in parallel
async def call_reflection(state: ReflectionState):
    await general_reflection_graph.ainvoke(state)


multi_reflection_graph = StateGraph(MultiMemoryInput)
multi_reflection_graph.add_node(determine_what_to_update)
multi_reflection_graph.add_node("reflection", call_reflection)
multi_reflection_graph.add_edge(START, "determine_what_to_update")
multi_reflection_graph = multi_reflection_graph.compile()

EMAIL_WRITING_INSTRUCTIONS = """As {name}'s email assistant: Help maintain their reputation as a capable final-year student balancing academics, career, and social connections.

{background}

# Question tool
Get required info before drafting. Never use placeholders. Ask {name} directly about:
- Job/internship details
- Event availability
- Interest in opportunities
- Academic status
- Current commitments
- Sender relationships
Never accept commitments without approval. Use MeetingAssistant for availability.

# ResponseEmailDraft tool
Write as {name}. Adapt tone:
- Academic: Professional, precise, engaged
- Professional: Confident, achievement-oriented
- Networking: Appreciative, purposeful
- Administrative: Clear, procedural
- Collaborative: Reliable, contributive
- Social: Personable, authentic

Strategies:
1. Academic: Honest, solution-oriented, specific
2. Career: Skills-focused, company-aware, concrete
3. Networking: Concise, genuine, action-oriented
4. Administrative: Procedure-aware, detailed

{response_preferences}

# SendCalendarInvite tool
Schedule only when certain and calendar is free. Priorities:
1. Job interviews/career
2. Academic requirements
3. Networking
4. Project coordination
5. Academic advising
6. Organizational duties
7. Social events

Guidelines:
- Interviews: Include prep buffer
- Academic: Align with office hours
- Social: Evenings/weekends
- Projects: Include pre-deadline buffer

{schedule_preferences}

# NewEmailDraft tool
For initiating conversations about:
- Job/internship follow-ups
- Networking
- References
- Opportunities
- Introductions
- Research
- Academic clarifications
- Resources
- Event coordination

Include: Subject, greeting, purpose, requests, closing, signature

# MeetingAssistant tool
ONLY use with EXPLICIT approval or confirmed meeting requests. Default to text first.

Common meetings:
- Interviews
- Career discussions
- Academic advising
- Project coordination
- Networking
- Organizational duties

# Context
{name}'s priorities:
1. Post-graduation employment
2. Academic excellence
3. Professional networking
4. Campus involvement
5. Graduation requirements

Project: Competence, professionalism, academic engagement, organization, authenticity

{random_preferences}"""

draft_prompt = """{instructions}

Respond with JSON for one tool:
1. Question: {{"name": "Question", "arguments": {{"content": "question"}}}}
2. ResponseEmailDraft: {{"name": "ResponseEmailDraft", "arguments": {{"content": "email", "new_recipients": []}}}}
3. NewEmailDraft: {{"name": "NewEmailDraft", "arguments": {{"content": "email", "recipients": []}}}}
4. MeetingAssistant: {{"name": "MeetingAssistant", "arguments": {{"call": true}}}}
5. SendCalendarInvite: {{"name": "SendCalendarInvite", "arguments": {{"emails": [], "title": "title", "start_time": "time", "end_time": "time"}}}}
6. Ignore: {{"name": "Ignore", "arguments": {{"ignore": true}}}}

Email thread:
{email}"""
