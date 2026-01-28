"""
Chat Service
Handles AI Assistant conversations with RAG integration and function calling
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import uuid
from datetime import datetime
import json

from langdetect import detect
import httpx

from app.services.ai.provider_manager import ProviderManager, AIProvider
from app.services.ai.rag_engine import RAGEngine
from app.services.geophysics.function_registry import FunctionRegistry
from app.services.geophysics.processing_engine import ProcessingEngine
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChatService:
    """
    Chat service with RAG and function calling capabilities
    """
    
    def __init__(self):
        self.provider_manager = ProviderManager()
        self.rag_engine = RAGEngine()
        self.function_registry = FunctionRegistry()
        self.processing_engine = ProcessingEngine()
        self.conversations: Dict[str, List[Dict]] = {}
    
    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user message with RAG and function calling
        
        Args:
            message: User message
            conversation_id: Conversation ID (creates new if None)
            use_rag: Whether to use RAG for context
            context: Additional context (current data, project, etc.)
        
        Returns:
            Response with message, sources, function calls, etc.
        """
        try:
            # Detect language
            language = self._detect_language(message)
            
            # Create or get conversation
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # Add user message to history
            self.conversations[conversation_id].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # RAG: Search for relevant documents
            sources = []
            rag_context = ""
            if use_rag:
                rag_results = await self.rag_engine.search(message, top_k=settings.TOP_K_RESULTS)
                sources = rag_results
                
                if rag_results:
                    rag_context = "\n\n**Relevant Scientific Literature:**\n"
                    for i, result in enumerate(rag_results, 1):
                        rag_context += f"\n[{i}] {result['content']}\n"
                        rag_context += f"*Source: {result['citation']}*\n"
            
            # Function calling: Check if message is a processing command
            function_calls = []
            functions_to_execute = await self._interpret_as_function_call(message, context)
            
            if functions_to_execute:
                function_calls = functions_to_execute
            
            # Build system prompt
            system_prompt = self._build_system_prompt(language, rag_context, context)
            
            # Get AI response
            ai_response = await self._get_ai_response(
                message=message,
                system_prompt=system_prompt,
                conversation_history=self.conversations[conversation_id][:-1],
                functions=self.function_registry.get_function_schemas() if not functions_to_execute else None
            )
            
            # Execute function calls if requested by AI
            if ai_response.get("function_call"):
                function_result = await self._execute_function(
                    ai_response["function_call"],
                    context
                )
                function_calls.append(function_result)
                
                # Get final response after function execution
                ai_response = await self._get_ai_response(
                    message=f"Function executed successfully. Result: {function_result}",
                    system_prompt=system_prompt,
                    conversation_history=self.conversations[conversation_id]
                )
            
            # Add assistant response to history
            self.conversations[conversation_id].append({
                "role": "assistant",
                "content": ai_response["content"],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "message": ai_response["content"],
                "conversation_id": conversation_id,
                "sources": sources,
                "function_calls": function_calls,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            raise
    
    async def stream_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response for real-time updates
        """
        # For now, return complete response
        # TODO: Implement actual streaming
        response = await self.process_message(message, conversation_id, use_rag, context)
        yield response
    
    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            return detect(text)
        except:
            return "en"
    
    def _build_system_prompt(
        self,
        language: str,
        rag_context: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build system prompt based on language and context
        """
        language_instructions = {
            "pt": "Você é o GeoBot, um assistente especializado em geofísica. Responda SEMPRE em português brasileiro.",
            "en": "You are GeoBot, a specialized geophysics assistant. Always respond in English.",
            "es": "Eres GeoBot, un asistente especializado en geofísica. Responde SIEMPRE en español.",
        }
        
        lang_instruction = language_instructions.get(language, language_instructions["en"])
        
        prompt = f"""{lang_instruction}

You are an expert in:
- Gravity and magnetic geophysical methods
- Data processing and interpretation
- Geophysical software and tools

When citing scientific literature, always format citations properly.

When executing processing functions, explain what you're doing clearly.

{rag_context}

Current Context:
{json.dumps(context or {}, indent=2)}
"""
        
        return prompt
    
    async def _get_ai_response(
        self,
        message: str,
        system_prompt: str,
        conversation_history: List[Dict],
        functions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Get response from AI provider
        """
        config = await self.provider_manager.get_configuration()
        
        if not config:
            raise ValueError("AI provider not configured")
        
        provider = config["provider"]
        
        if provider == AIProvider.GROQ:
            return await self._call_groq(message, system_prompt, conversation_history, functions, config)
        elif provider == AIProvider.OPENAI:
            return await self._call_openai(message, system_prompt, conversation_history, functions, config)
        elif provider == AIProvider.CLAUDE:
            return await self._call_claude(message, system_prompt, conversation_history, functions, config)
        elif provider == AIProvider.GEMINI:
            return await self._call_gemini(message, system_prompt, conversation_history, functions, config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _call_groq(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict],
        functions: Optional[List[Dict]],
        config: Dict
    ) -> Dict[str, Any]:
        """
        Call Groq API using official SDK with streaming support
        
        Uses the official Groq Python SDK for reliable API access.
        Supports streaming and reasoning_effort parameter.
        """
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("groq package required. Install with: pip install groq")
        
        models = [config.get("model", "openai/gpt-oss-120b")] + [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        
        # Initialize Groq client
        client = Groq(api_key=config['api_key'])
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        
        for model in models:
            try:
                # Call with official SDK (non-streaming for now)
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=config.get("temperature", 1.0),
                    max_completion_tokens=config.get("max_completion_tokens", 8192),
                    top_p=config.get("top_p", 1.0),
                    reasoning_effort=config.get("reasoning_effort", "medium"),
                    stream=False,  # Can be enabled for streaming
                    stop=None
                )
                
                # Extract response
                choice = completion.choices[0]
                result = {"content": choice.message.content}
                
                # Handle function calls if supported
                if hasattr(choice.message, 'function_call') and choice.message.function_call:
                    result["function_call"] = {
                        "name": choice.message.function_call.name,
                        "arguments": choice.message.function_call.arguments
                    }
                
                logger.info(f"✅ Groq response received from model: {model}")
                return result
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error with Groq model {model}: {error_msg}")
                
                # Check for rate limit
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    logger.warning(f"⚠️ Rate limit for {model}, trying fallback...")
                    continue
                
                # If last model, raise error
                if model == models[-1]:
                    raise Exception(f"All Groq models failed. Last error: {error_msg}")
                
                continue
        
        raise Exception("All Groq models failed")
    
    async def _call_openai(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict],
        functions: Optional[List[Dict]],
        config: Dict
    ) -> Dict[str, Any]:
        """
        Call OpenAI API (GPT-4, GPT-3.5-turbo, etc.)
        
        Supports function calling and streaming
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package required. Install with: pip install openai")
        
        try:
            client = AsyncOpenAI(api_key=config["api_key"])
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})
            
            # Prepare request parameters
            params = {
                "model": config["model"],
                "messages": messages,
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_tokens", 4096)
            }
            
            # Add functions if available
            if functions:
                params["functions"] = functions
                params["function_call"] = "auto"
            
            # Make API call
            response = await client.chat.completions.create(**params)
            
            choice = response.choices[0]
            
            result = {
                "content": choice.message.content or "",
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Handle function calls
            if choice.message.function_call:
                result["function_call"] = {
                    "name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments
                }
            
            logger.info(f"OpenAI response: {response.model} - {result['usage']['total_tokens']} tokens")
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _call_claude(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict],
        functions: Optional[List[Dict]],
        config: Dict
    ) -> Dict[str, Any]:
        """
        Call Claude API (Claude 3 Opus, Sonnet, Haiku)
        
        Note: Claude uses different approach for function calling
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise RuntimeError("anthropic package required. Install with: pip install anthropic")
        
        try:
            client = AsyncAnthropic(api_key=config["api_key"])
            
            # Claude requires different message format (no system in messages)
            messages = []
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            messages.append({"role": "user", "content": message})
            
            # Prepare request parameters
            params = {
                "model": config["model"],
                "system": system_prompt,  # Claude uses separate system parameter
                "messages": messages,
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_tokens", 4096)
            }
            
            # Claude 3 has experimental tool use (similar to function calling)
            if functions:
                # Convert OpenAI function format to Claude tool format
                tools = []
                for func in functions:
                    tool = {
                        "name": func["name"],
                        "description": func["description"],
                        "input_schema": {
                            "type": "object",
                            "properties": func["parameters"]["properties"],
                            "required": func["parameters"].get("required", [])
                        }
                    }
                    tools.append(tool)
                params["tools"] = tools
            
            # Make API call
            response = await client.messages.create(**params)
            
            # Extract text content
            content = ""
            function_call = None
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    # Claude's function calling
                    function_call = {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
            
            result = {
                "content": content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
            if function_call:
                result["function_call"] = function_call
            
            logger.info(f"Claude response: {response.model} - {result['usage']['total_tokens']} tokens")
            
            return result
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise Exception(f"Claude API error: {str(e)}")
    
    async def _call_gemini(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict],
        functions: Optional[List[Dict]],
        config: Dict
    ) -> Dict[str, Any]:
        """
        Call Google Gemini API (Gemini Pro, Gemini Pro Vision)
        
        Note: Gemini has different API structure than OpenAI
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("google-generativeai package required. Install with: pip install google-generativeai")
        
        try:
            genai.configure(api_key=config["api_key"])
            
            # Get model
            model = genai.GenerativeModel(config["model"])
            
            # Build conversation history
            # Gemini uses a chat format
            chat_history = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
            
            # Start chat with history
            chat = model.start_chat(history=chat_history)
            
            # Combine system prompt with user message
            full_message = f"{system_prompt}\n\n{message}"
            
            # Generate response
            response = await chat.send_message_async(
                full_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.get("temperature", 0.7),
                    max_output_tokens=config.get("max_tokens", 4096),
                )
            )
            
            # Extract content
            content = response.text
            
            result = {
                "content": content,
                "model": config["model"],
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                }
            }
            
            # Note: Gemini function calling is in preview and has different syntax
            # For now, we'll use semantic interpretation for functions
            if functions:
                logger.warning("Gemini function calling not fully implemented yet")
            
            logger.info(f"Gemini response: {config['model']} - {result['usage']['total_tokens']} tokens")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def _interpret_as_function_call(
        self,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Interpret message as processing function call
        """
        # Use semantic similarity to match message to functions
        functions = self.function_registry.search_functions(message)
        
        if functions:
            return functions
        
        return []
    
    async def _execute_function(
        self,
        function_call: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute processing function
        """
        function_name = function_call["name"]
        parameters = json.loads(function_call["arguments"])
        
        # Execute via processing engine
        result = await self.processing_engine.execute(
            function_name=function_name,
            data_id=context.get("data_id") if context else None,
            parameters=parameters
        )
        
        return result
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    async def delete_conversation(self, conversation_id: str):
        """Delete conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
