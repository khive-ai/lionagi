#!/usr/bin/env python3
"""
Claude Code Proxy API Server - OpenAI Compatible
A FastAPI server that proxies requests to Claude Code CLI using OpenAI API format.
Compatible with Roo Code, Cline, and other OpenAI API clients.
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
import uuid
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# FastAPI app
app = FastAPI(
    title="Claude Code Proxy API - OpenAI Compatible",
    description="OpenAI-compatible proxy server for Claude Code CLI - Personal Development Use",
    version="2.0.0"
)

# Request/Response Models matching OpenAI API
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None

# Configuration
CLAUDE_CODE_BINARY = shutil.which("claude") or "claude"
WORKING_DIR = Path.home() / ".claude-proxy" / "workspaces"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# Check if Claude Code is available
def check_claude_code():
    """Verify Claude Code CLI is installed and accessible"""
    try:
        result = subprocess.run(
            [CLAUDE_CODE_BINARY, "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Claude Code not found at {CLAUDE_CODE_BINARY}")
        logger.info(f"Claude Code found: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Claude Code check failed: {e}")
        raise RuntimeError(
            "Claude Code CLI not found. Please install it first:\n"
            "npm install -g @anthropic-ai/claude-code"
        )

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    check_claude_code()
    logger.info("Claude Code Proxy API (OpenAI Compatible) started successfully")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "claude-code-proxy-openai"}

# Main chat completions endpoint (OpenAI compatible)
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Main endpoint that proxies to Claude Code CLI - OpenAI compatible"""
    
    # Extract the conversation
    conversation = []
    system_prompt = None
    
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            conversation.append(f"Human: {msg.content}")
        elif msg.role == "assistant":
            conversation.append(f"Assistant: {msg.content}")
    
    # Build the prompt in Claude format
    full_prompt = ""
    if system_prompt:
        full_prompt = f"System: {system_prompt}\n\n"
    
    full_prompt += "\n\n".join(conversation)
    
    # Ensure we end with Human/Assistant pattern
    if conversation and conversation[-1].startswith("Human:"):
        full_prompt += "\n\nAssistant:"
    
    logger.info(f"Processing request for model: {request.model}")
    
    # Create a unique workspace for this request
    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    workspace = WORKING_DIR / request_id
    workspace.mkdir(exist_ok=True)
    
    try:
        # Execute Claude Code
        cmd = [
            CLAUDE_CODE_BINARY,
            "-p", full_prompt,
            "--output-format", "stream-json" if request.stream else "json",
            "--max-turns", "1"
        ]
        
        logger.debug(f"Executing command: {' '.join(cmd[:4])}...")
        

        # Non-streaming response
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace)
        )
        
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            logger.error(f"Claude Code error: {stderr.decode()}")
            raise HTTPException(
                status_code=500,
                detail=f"Claude Code execution failed: {stderr.decode()}"
            )
        
        # Parse response
        try:
            claude_response = json.loads(stdout.decode())
            
            return claude_response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            # Fallback to text response
            return format_text_response(stdout.decode(), request_id, request.model)
                
    finally:
        # Cleanup workspace
        try:
            shutil.rmtree(workspace)
        except:
            pass

def format_text_response(text: str, request_id: str, model: str) -> ChatCompletionResponse:
    """Format plain text response in OpenAI format"""
    content = text.strip()
    prompt_tokens = len(content.split()) * 2
    completion_tokens = len(content.split())
    
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return Response(
        content=json.dumps({
            "error": {
                "message": str(exc),
                "type": "internal_error",
                "param": None,
                "code": None
            }
        }),
        status_code=500,
        media_type="application/json"
    )

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Claude Code Proxy API Server - OpenAI Compatible")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Run server
    uvicorn.run(
        "claude_code_proxy:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )