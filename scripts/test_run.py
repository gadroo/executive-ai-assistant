import asyncio
from langgraph_sdk import get_client
import uuid
import hashlib

# Test email data
TEST_EMAIL = {
    "id": "test_id_12345",
    "thread_id": "test_thread_12345",
    "from_email": "sender@example.com",
    "to_email": "recipient@example.com",
    "subject": "Test Email Subject",
    "page_content": "This is a test email body. Please process this test message.",
    "send_time": "2025-04-29T14:00:00"
}

async def main():
    # Connect to langgraph server
    client = get_client(url="http://127.0.0.1:2024")
    
    # Generate a deterministic thread ID from the thread_id
    thread_id = str(
        uuid.UUID(hex=hashlib.md5(TEST_EMAIL["thread_id"].encode("UTF-8")).hexdigest())
    )
    
    # Create a thread if it doesn't exist
    try:
        await client.threads.get(thread_id)
        print(f"Thread {thread_id} already exists")
    except Exception:
        print(f"Creating thread {thread_id}")
        await client.threads.create(thread_id=thread_id)
    
    # Update thread metadata
    await client.threads.update(thread_id, metadata={"email_id": TEST_EMAIL["id"]})
    
    # Define the full initial state matching the State TypedDict
    initial_state = {
        "email": TEST_EMAIL,
        "triage": None,  # Initialize triage as None
        "messages": []    # Initialize messages as an empty list
    }
    
    # Create a run with the full initial state as input
    print(f"Starting run with test email: {TEST_EMAIL['subject']}")
    # The input needs to match the State structure expected by the entry node ('email' key)
    run = await client.runs.create(
        thread_id,
        "main",
        input=initial_state, 
        multitask_strategy="rollback",
    )
    
    print(f"Run created with ID: {run['id']}")
    print("Check your langgraph server logs to see the progress")

if __name__ == "__main__":
    asyncio.run(main()) 