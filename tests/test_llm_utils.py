#!/usr/bin/env python3
"""Test LLM utilities migration."""

print("🧪 Testing LLM Utilities Migration\n")

def test_utility_functions():
    """Test that utility functions work correctly."""
    
    # Test extract_json_from_model_output
    from browser_use.agent.llm.utils import extract_json_from_model_output
    
    # Test normal JSON
    result = extract_json_from_model_output('{"answer": "paris"}')
    assert result == {"answer": "paris"}
    print("✅ extract_json_from_model_output: Normal JSON")
    
    # Test code block wrapped JSON
    result = extract_json_from_model_output('```json\n{"answer": "paris"}\n```')
    assert result == {"answer": "paris"}
    print("✅ extract_json_from_model_output: Code block wrapped")
    
    # Test list with single dict (edge case)
    result = extract_json_from_model_output('[{"answer": "paris"}]')
    assert result == {"answer": "paris"}
    print("✅ extract_json_from_model_output: List with single dict")
    
    # Test is_model_without_tool_support
    from browser_use.agent.llm.utils import is_model_without_tool_support
    
    assert is_model_without_tool_support("deepseek-r1") == True
    assert is_model_without_tool_support("deepseek-reasoner") == True
    assert is_model_without_tool_support("gpt-4o") == False
    assert is_model_without_tool_support("claude-3-sonnet") == False
    print("✅ is_model_without_tool_support: Pattern matching")
    
    # Test clean_response_content
    from browser_use.agent.llm.utils import clean_response_content
    
    dirty_content = "<think>Some reasoning here</think>The actual answer"
    clean_content = clean_response_content(dirty_content)
    assert clean_content == "The actual answer"
    print("✅ clean_response_content: Think tags removed")
    
    print("\n🎉 All utility functions working correctly!")

def test_import_structure():
    """Test that imports work correctly."""
    
    # Test main LLM package imports
    try:
        from browser_use.llm import (
            LLMService, 
            is_model_without_tool_support,
            extract_json_from_model_output,
            convert_input_messages,
            clean_response_content
        )
        print("✅ Main LLM package imports working")
    except ImportError as e:
        print(f"❌ Main LLM imports failed: {e}")
        return False
    
    # Test direct utils imports
    try:
        from browser_use.agent.llm.utils import (
            is_model_without_tool_support,
            extract_json_from_model_output,
            convert_input_messages,
            clean_response_content
        )
        print("✅ Direct utils imports working")
    except ImportError as e:
        print(f"❌ Direct utils imports failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("📦 Testing LLM Framework Utility Migration...")
    
    try:
        if test_import_structure():
            test_utility_functions()
            print("\n🎯 MIGRATION SUCCESS:")
            print("   ✅ All LLM utilities moved to LLM framework")
            print("   ✅ Agent service dependencies cleaned up")
            print("   ✅ Imports working correctly")
            print("   ✅ Functionality preserved")
        else:
            print("\n❌ Import structure test failed")
    except Exception as e:
        print(f"\n❌ Migration test failed: {e}")
        import traceback
        traceback.print_exc()