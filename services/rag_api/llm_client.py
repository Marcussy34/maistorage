"""
LLM client for handling OpenAI chat completions in the MAI Storage RAG system.

This module provides a swappable interface for LLM interactions, supporting
different models and configurations as specified in Phase 2.5 of the plan.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import time
from datetime import datetime

import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    max_retries: int = 3


class LLMResponse(BaseModel):
    """Response from LLM generation."""
    
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: str
    response_time_ms: float
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class LLMClient:
    """
    OpenAI client wrapper for chat completions with swappable model support.
    
    Supports both synchronous and asynchronous operations, with automatic
    fallback handling and performance monitoring.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client with configuration.
        
        Args:
            config: LLM configuration, uses defaults if None
        """
        self.config = config or LLMConfig()
        
        # Set up OpenAI client
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_response_time = 0.0
        
        logger.info(f"LLM client initialized with model: {self.config.model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion synchronously.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()
        
        try:
            # Merge config defaults with provided kwargs
            completion_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            # Remove None values
            completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}
            
            # Make API call
            response = self.client.chat.completions.create(**completion_kwargs)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}
            finish_reason = response.choices[0].finish_reason
            
            # Update stats
            self.total_requests += 1
            self.total_tokens += usage.get("total_tokens", 0)
            self.total_response_time += response_time_ms
            
            logger.info(f"Chat completion: {usage.get('total_tokens', 0)} tokens in {response_time_ms:.2f}ms")
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def achat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion asynchronously.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()
        
        try:
            # Merge config defaults with provided kwargs
            completion_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            # Remove None values
            completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}
            
            # Make API call
            response = await self.async_client.chat.completions.create(**completion_kwargs)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}
            finish_reason = response.choices[0].finish_reason
            
            # Update stats
            self.total_requests += 1
            self.total_tokens += usage.get("total_tokens", 0)
            self.total_response_time += response_time_ms
            
            logger.info(f"Async chat completion: {usage.get('total_tokens', 0)} tokens in {response_time_ms:.2f}ms")
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            logger.error(f"Async chat completion failed: {e}")
            raise
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for OpenAI API
            
        Yields:
            Content chunks as they arrive
        """
        try:
            # Merge config defaults with provided kwargs
            completion_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
            
            # Remove None values
            completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}
            
            # Make streaming API call
            stream = await self.async_client.chat.completions.create(**completion_kwargs)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the LLM connection with a simple prompt.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "user", "content": "Say 'Hello' if you can read this."}
            ]
            
            response = self.chat_completion(
                messages=test_messages,
                max_tokens=10,
                temperature=0.0
            )
            
            return "hello" in response.content.lower()
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "model": self.config.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_response_time_ms": round(avg_response_time, 2),
            "avg_tokens_per_request": (
                round(self.total_tokens / self.total_requests, 2)
                if self.total_requests > 0 else 0.0
            )
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated LLM config: {key} = {value}")


# Factory function for easy client creation
def create_llm_client(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client with common configurations.
    
    Args:
        model: OpenAI model name
        api_key: API key (uses env var if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LLMClient instance
    """
    config = LLMConfig(
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return LLMClient(config)
