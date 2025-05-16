"""Agent responsible for rewriting the email in a better tone."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from eaia.schemas import State, ReWriteEmail

from eaia.main.config import get_config_async


rewrite_prompt = """You job is to rewrite an email draft to sound more like {name}.

{name}'s assistant just drafted an email. It is factually correct, but it may not sound like {name}. \
Your job is to rewrite the email keeping the information the same (do not add anything that is made up!) \
but adjusting the tone to match {name}'s authentic voice and writing style.

{instructions}

EXAMPLES:

Example 1 - Professional Email:
<original_draft>
Dear Professor Santosh,

I am writing to express my interest in your AI research lab. I have been following your work on reinforcement learning and would be eager to contribute. My experience with TensorFlow and PyTorch would be relevant to your current projects.

Thank you for your time.
Sincerely,
Aryan
</original_draft>

<rewritten>
Hi Professor Santosh,

I've been following your recent publications on reinforcement learning techniques and I'm particularly interested in your lab's approach to policy optimization. Having implemented several RL models using both TensorFlow and PyTorch in my ReflectAI project, I believe I could contribute meaningfully to your research.

Best regards,
Aryan
</rewritten>

Example 2 - Casual Communication:
<original_draft>
Hello Team,

Regarding our project timeline, I think we should adjust our milestones. The current schedule doesn't account for integration testing properly. I propose we add an additional week before the final delivery.

Thank you,
Aryan
</original_draft>

<rewritten>
Hey team,

Looking at our timeline, we're going to need to shift things around a bit. The integration testing phase is too compressed - let's add another week before delivery to make sure everything works properly with the ElasticSearch implementation.

- Aryan
</rewritten>

Here is the assistant's current draft:

<draft>
{draft}
</draft>

Here is the email thread:

From: {author}
To: {to}
Subject: {subject}

{email_thread}"""


async def rewrite(state: State, config, store):
    model = config["configurable"].get("model", "claude-3-5-sonnet-latest")
    llm = ChatAnthropic(model=model, temperature=0)
    prev_message = state["messages"][-1]
    draft = prev_message.tool_calls[0]["args"]["content"]
    namespace = (config["configurable"].get("assistant_id", "default"),)
    result = await store.aget(namespace, "rewrite_instructions")
    prompt_config = await get_config_async(config)
    if result and "data" in result.value:
        _prompt = result.value["data"]
    else:
        await store.aput(
            namespace,
            "rewrite_instructions",
            {"data": prompt_config["rewrite_preferences"]},
        )
        _prompt = prompt_config["rewrite_preferences"]
    input_message = rewrite_prompt.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        subject=state["email"]["subject"],
        to=state["email"]["to_email"],
        draft=draft,
        instructions=_prompt,
        name=prompt_config["name"],
    )
    # Configure model with tools in a way compatible with Claude
    llm_with_tools = llm.bind_tools([ReWriteEmail])
    
    # Call with input as a message object
    messages = [HumanMessage(content=input_message)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract the response data
    tool_call = response.tool_calls[0]
    tool_call_response = ReWriteEmail.model_validate(tool_call["args"])
    rewritten_content = tool_call_response.rewritten_content
    
    tool_calls = [
        {
            "id": prev_message.tool_calls[0]["id"],
            "name": prev_message.tool_calls[0]["name"],
            "args": {
                **prev_message.tool_calls[0]["args"],
                **{"content": rewritten_content},
            },
        }
    ]
    prev_message = {
        "role": "assistant",
        "id": prev_message.id,
        "content": prev_message.content,
        "tool_calls": tool_calls,
    }
    return {"messages": [prev_message]}
