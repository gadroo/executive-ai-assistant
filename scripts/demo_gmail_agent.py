#!/usr/bin/env python
"""
Demo script for the Gmail agent with Anthropic integration.
This script shows how to use the Gmail draft agent to create drafts
and search for existing drafts in your Gmail account.
"""

import os
import argparse
import logging
from dotenv import load_dotenv

from eaia.gmail_draft_agent import draft_email, search_drafts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Check if the required API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        print("Please set the ANTHROPIC_API_KEY environment variable")
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gmail Agent Demo with Anthropic")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'draft' command for creating drafts
    draft_parser = subparsers.add_parser("draft", help="Create a draft email")
    draft_parser.add_argument("prompt", help="Prompt describing the email to draft")
    
    # 'search' command for searching drafts
    search_parser = subparsers.add_parser("search", help="Search for drafts")
    search_parser.add_argument("query", help="Search query to find drafts")
    
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "draft":
        logger.info(f"Creating a draft email based on prompt: {args.prompt}")
        response = draft_email(args.prompt)
        print("\nAgent Response:")
        print(response)
    
    elif args.command == "search":
        logger.info(f"Searching for drafts with query: {args.query}")
        response = search_drafts(args.query)
        print("\nSearch Results:")
        print(response)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 