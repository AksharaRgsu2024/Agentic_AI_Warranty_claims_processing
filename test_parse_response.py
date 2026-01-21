#!/usr/bin/env python3
"""Test the _parse_response method directly"""

import json
import logging
from langchain_core.messages import AIMessage

logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

# Simulate AIMessage responses with JSON in different formats
test_responses = [
    # Format 1: Plain JSON with markdown code block
    AIMessage(content='''Based on the claim details, here's my analysis:

```json
{
  "claim_validity": "Valid",
  "warranty_coverage": "Covered",
  "decision": "Approve Claim",
  "confidence_score": 0.95,
  "reasons": ["Product is within warranty period", "Issue is covered by policy"],
  "next_steps": ["Send approval email", "Arrange replacement"],
  "notes": "Clear warranty coverage"
}
```

This claim meets all requirements for approval.'''),
    
    # Format 2: JSON without code block markers
    AIMessage(content='''Analysis complete. Here is the decision:

{
  "claim_validity": "Invalid",
  "warranty_coverage": "Not Covered",
  "decision": "Reject Claim",
  "confidence_score": 0.85,
  "reasons": ["Product is out of warranty", "Damage not covered"],
  "next_steps": ["Notify customer", "Close claim"],
  "notes": "No coverage available"
}

The customer should be informed promptly.'''),
]

# Test parsing
class FakeRAGAgent:
    def _parse_response(self, response, model_class=None):
        """Parse response from LLM - handles both structured output and raw JSON"""
        try:
            import json
            import re
            from langchain_core.messages import BaseMessage
            
            # Check for AIMessage/BaseMessage first - these have content that needs parsing
            if isinstance(response, BaseMessage):
                content = response.content
                logging.debug(f"_parse_response: Processing BaseMessage content (length: {len(content)})")
                logging.debug(f"Response content preview: {content[:200]}")
                
                # Try to extract JSON from markdown code blocks
                # Pattern 1: ```json ... ``` with flexible whitespace
                json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    logging.debug(f"Found markdown code block, attempting to parse JSON")
                    try:
                        parsed = json.loads(json_str)
                        logging.info(f"Successfully parsed JSON from markdown code block")
                        return parsed
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse JSON from markdown block: {e}")
                
                # Pattern 2: Try to find any JSON object/array in the content
                # Look for content that starts with { or [
                json_start = content.find('{')
                if json_start >= 0:
                    logging.debug(f"Found {{ at position {json_start}, attempting brace-matching extraction")
                    # Try to extract from the first { to a closing }
                    try:
                        # Find matching closing brace
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        
                        for i, char in enumerate(content[json_start:], start=json_start):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_str = content[json_start:i+1]
                                        parsed = json.loads(json_str)
                                        logging.info(f"Successfully parsed JSON using brace-matching (found {len(parsed)} fields)")
                                        return parsed
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.debug(f"Failed to extract JSON using brace-matching: {e}")
                
                # If no JSON found, log the full content for debugging
                logging.error(f"Could not parse JSON from BaseMessage content. Full content:\n{content}")
                return {"raw_response": content[:500]}
            
            # If it's already a Pydantic model instance, convert to dict
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'dict') and not isinstance(response, str):
                return response.dict()
            # If it's a string, try to parse as JSON
            elif isinstance(response, str):
                return json.loads(response)
            else:
                # Try to convert to dict if possible
                if hasattr(response, '__dict__'):
                    return response.__dict__
                return response
        except Exception as e:
            logging.error(f"Error parsing response: {e}. Response type: {type(response)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "response_type": str(type(response))}

# Test
agent = FakeRAGAgent()

print("="*80)
print("Testing JSON Parsing")
print("="*80)

for i, test_response in enumerate(test_responses, 1):
    print(f"\n--- Test {i} ---")
    result = agent._parse_response(test_response)
    print(f"Result keys: {list(result.keys())}")
    print(f"Decision: {result.get('decision', 'N/A')}")
    print(f"Confidence: {result.get('confidence_score', 'N/A')}")
    print()

print("="*80)
print("✓ Test completed")
print("="*80)
