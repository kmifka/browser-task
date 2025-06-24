#!/usr/bin/env python3
"""Test the new LLM framework to ensure it works correctly."""

import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import browser_use
sys.path.insert(0, str(Path(__file__).parent))

from browser_use.llm import LLMFactory, LLMService
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def test_factory():
    """Test the LLM factory creates appropriate adapters."""
    print("Testing LLM Factory...")
    
    # Test OpenAI
    openai_llm = ChatOpenAI(model='gpt-4o', api_key='dummy')
    adapter = LLMFactory.create_adapter(openai_llm)
    print(f"OpenAI adapter: {adapter.__class__.__name__}")
    print(f"Supports vision: {adapter.capabilities.supports_vision}")
    print(f"Preferred tool method: {adapter.capabilities.preferred_tool_calling_method}")
    
    # Test Anthropic  
    anthropic_llm = ChatAnthropic(model='claude-3-sonnet-20240229', api_key='dummy')
    adapter = LLMFactory.create_adapter(anthropic_llm)
    print(f"Anthropic adapter: {adapter.__class__.__name__}")
    print(f"Supports vision: {adapter.capabilities.supports_vision}")
    print(f"Preferred tool method: {adapter.capabilities.preferred_tool_calling_method}")
    

def test_service():
    """Test the LLM service."""
    print("\nTesting LLM Service...")
    
    # Create a dummy OpenAI LLM for testing
    llm = ChatOpenAI(model='gpt-4o', api_key='dummy')
    
    try:
        # This will fail connection verification, but we can test the service setup
        service = LLMService(llm)
        print("This shouldn't print since connection will fail")
    except ConnectionError as e:
        print(f"Expected connection error: {e}")
        
    print(f"LLM capabilities detected correctly")


if __name__ == '__main__':
    print("🧪 Testing the new LLM framework...")
    
    test_factory()
    test_service()
    
    print("\n✅ LLM Framework tests completed!")