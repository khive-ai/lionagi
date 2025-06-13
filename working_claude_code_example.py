#!/usr/bin/env python3
"""
Working example of Claude Code with lionagi using direct endpoint calls.

Since there are integration issues with Branch/iModel, this example
shows how to use Claude Code endpoint directly while still benefiting
from lionagi's infrastructure.
"""

import asyncio
from lionagi.service.providers.anthropic_.claude_code import ClaudeCodeEndpoint
from lionagi.session.branch import Branch
from lionagi.protocols.messages.assistant_response import AssistantResponse


class ClaudeCodeBranch:
    """A simple wrapper that combines Branch with direct Claude Code calls."""
    
    def __init__(self, name: str, allowed_tools: list[str] = None):
        self.name = name
        self.branch = Branch(name=name)
        self.endpoint = ClaudeCodeEndpoint()
        self.allowed_tools = allowed_tools or ["Read", "Write", "Edit", "Bash"]
        self.session_id = None
    
    async def chat(self, message: str, **kwargs) -> str:
        """Send a message to Claude Code and track the conversation."""
        
        # Build the request
        request = {
            "messages": [{"role": "user", "content": message}],
            "allowed_tools": kwargs.get("allowed_tools", self.allowed_tools),
        }
        
        # Add session_id if we have one (for resuming)
        if self.session_id and "new_session" not in kwargs:
            request["session_id"] = self.session_id
            print(f"  [Resuming session: {self.session_id[:8]}...]")
        
        # Add any other parameters
        for key in ["multi_turn", "max_turns", "model"]:
            if key in kwargs:
                request[key] = kwargs[key]
        
        # Make the API call
        response = await self.endpoint.call(request)
        
        # Update session ID
        if isinstance(response, dict) and "session_id" in response:
            new_session = response["session_id"]
            if new_session != self.session_id:
                self.session_id = new_session
                print(f"  [Session updated: {new_session[:8]}...]")
        
        # Store the conversation in the branch
        self.branch.messages.create_message(
            role="user",
            content=message
        )
        
        # Create assistant response
        assistant_msg = AssistantResponse.create(
            assistant_response=response,
            metadata={"model_response": response}
        )
        self.branch.messages.add(assistant_msg)
        
        # Return the content
        return response.get("content", response.get("result", ""))
    
    def new_session(self):
        """Start a new session, forgetting the previous one."""
        self.session_id = None
        print(f"  [Starting new session]")


async def example_basic():
    """Basic example with session continuity."""
    print("=== Basic Claude Code Example ===\n")
    
    # Create a Claude Code branch
    assistant = ClaudeCodeBranch("Dev Assistant", allowed_tools=["Write", "Read", "LS"])
    
    # First interaction
    print("1. Creating a file:")
    response1 = await assistant.chat(
        "Create a file called 'greeting.txt' with the message 'Hello from Claude Code!'"
    )
    print(f"Response: {response1}\n")
    
    # Second interaction (resumes session)
    print("2. Checking our work:")
    response2 = await assistant.chat(
        "List the files in the current directory and show the contents of greeting.txt"
    )
    print(f"Response: {response2}\n")
    
    # Third interaction (still same session)
    print("3. Making changes:")
    response3 = await assistant.chat(
        "Add a second line to greeting.txt that says 'This is a test file.'"
    )
    print(f"Response: {response3}\n")
    
    # Start a new session
    assistant.new_session()
    
    print("4. New session - different context:")
    response4 = await assistant.chat(
        "What files have we created? (This is a new session, so I shouldn't remember)"
    )
    print(f"Response: {response4}\n")
    
    # Clean up
    print("5. Cleanup:")
    response5 = await assistant.chat(
        "Remove greeting.txt if it exists",
        allowed_tools=["Bash", "LS"]
    )
    print(f"Response: {response5}\n")


async def example_multi_branch():
    """Example with multiple Claude Code branches."""
    print("=== Multi-Branch Example ===\n")
    
    # Create specialized assistants
    frontend = ClaudeCodeBranch("Frontend Dev", ["Write", "Read", "Edit"])
    backend = ClaudeCodeBranch("Backend Dev", ["Write", "Read", "Edit", "Bash"])
    tester = ClaudeCodeBranch("QA Engineer", ["Read", "Write", "Bash"])
    
    # Frontend creates a component
    print("1. Frontend creating component:")
    await frontend.chat(
        "Create a simple HTML file called 'index.html' with a button that says 'Click me!'",
        multi_turn=True,
        max_turns=2
    )
    
    # Backend creates an API
    print("\n2. Backend creating server:")
    await backend.chat(
        "Create a simple Node.js server in 'server.js' that serves the index.html file",
        multi_turn=True,
        max_turns=2
    )
    
    # Tester writes tests
    print("\n3. QA creating tests:")
    await tester.chat(
        "Create a simple test script 'test.sh' that checks if server.js and index.html exist",
        multi_turn=True,
        max_turns=2
    )
    
    # Frontend makes updates (resume session)
    print("\n4. Frontend updating (resume session):")
    await frontend.chat(
        "Add some CSS styling to the button in index.html"
    )
    
    # Show all sessions
    print("\n=== Session Summary ===")
    print(f"Frontend session: {frontend.session_id[:8] if frontend.session_id else 'None'}...")
    print(f"Backend session: {backend.session_id[:8] if backend.session_id else 'None'}...")
    print(f"QA session: {tester.session_id[:8] if tester.session_id else 'None'}...")
    
    # Cleanup
    print("\n5. Cleanup all files:")
    await backend.chat("Remove index.html, server.js, and test.sh if they exist")


async def example_multi_turn():
    """Example showing multi-turn capabilities."""
    print("=== Multi-Turn Example ===\n")
    
    assistant = ClaudeCodeBranch("Python Dev", ["Write", "Read", "Edit", "Bash"])
    
    print("Creating and testing a Python module:")
    response = await assistant.chat(
        "Create a Python module called 'math_utils.py' with factorial and fibonacci functions, "
        "then create a test file to test these functions, and run the tests",
        multi_turn=True,
        max_turns=5
    )
    print(f"Final result: {response}\n")
    
    # Follow up in the same session
    print("Adding more functionality:")
    response2 = await assistant.chat(
        "Add a prime number checker function to math_utils.py and update the tests"
    )
    print(f"Update result: {response2}\n")
    
    # Cleanup
    await assistant.chat("Remove math_utils.py and any test files")


async def main():
    """Run all examples."""
    print("Claude Code + lionagi Integration Examples\n")
    
    try:
        # Run basic example
        await example_basic()
        
        print("\n" + "="*60 + "\n")
        
        # Run multi-branch example
        await example_multi_branch()
        
        print("\n" + "="*60 + "\n")
        
        # Run multi-turn example
        await example_multi_turn()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())