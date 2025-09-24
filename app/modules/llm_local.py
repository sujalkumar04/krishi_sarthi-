# # in app/modules/llm_local.py

# import logging
# import os
# import re
# import platform
# import asyncio
# from typing import Dict, Any, List, Optional
# from datetime import datetime
# import json
# import multiprocessing as mp

# # (The _child_probe_for_metal function remains the same)
# def _child_probe_for_metal(model_path: str, gpu_layers_arg: int):
#     try:
#         os.environ.pop("LLAMA_NO_METAL", None)
#         from llama_cpp import Llama
#         llm = Llama(model_path=model_path, n_ctx=1024, n_threads=2, n_gpu_layers=gpu_layers_arg, n_batch=64, f16_kv=True, logits_all=False, use_mmap=True, use_mlock=False, chat_format="llama-3", verbose=False)
#         _ = llm.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.1)
#         os._exit(0)
#     except Exception:
#         os._exit(1)

# from config import config

# logger = logging.getLogger(__name__)

# class LocalLLMService:
#     """Local LLM service using LLaMA-8B via llama-cpp-python"""
    
#     def __init__(self):
#         self.model_path = config.LOCAL_LLM_PATH
#         self.llm = None
#         self.is_initialized = False
#         self.model_info = {}
#         self._gpu_layers: int = 35
        
#         self.prompt_templates = {
#             "hi": self._get_prompt_template(),
#             "en": self._get_prompt_template(),
#             "hinglish": self._get_prompt_template()
#         }
        
#         self.system_prompts = {
#             "irrigation": self._get_irrigation_system_prompt(),
#             "fertilizer": self._get_fertilizer_system_prompt(),
#             "pest_control": self._get_pest_control_system_prompt(),
#             "weather": self._get_weather_system_prompt(),
#             "harvest": self._get_harvest_system_prompt(),
#             "default": self._get_default_system_prompt()
#         }
    
#     # --- (initialize, _load_model_blocking, etc. remain the same as the working version) ---
#     async def initialize(self):
#         try:
#             if platform.system().lower() == "darwin" and os.environ.get("ENABLE_LOCAL_LLM", "0") != "1":
#                 logger.warning("Local LLM disabled by default on macOS. Set ENABLE_LOCAL_LLM=1 to enable.")
#                 self.is_initialized = False; return
#             if platform.system().lower() == "darwin":
#                 os.environ.setdefault("OMP_NUM_THREADS", str(mp.cpu_count() // 2))
#             if not self._check_model_file():
#                 logger.error(f"Model file not found: {self.model_path}"); self.is_initialized = False; return
#             gpu_safe = await asyncio.to_thread(self._probe_gpu_layers_supported, self._gpu_layers)
#             if not gpu_safe:
#                 logger.warning("Metal offload probe failed; falling back to CPU-only"); self._gpu_layers = 0; os.environ["LLAMA_NO_METAL"] = "1"
#             else:
#                 logger.info(f"Metal offload probe successful. Using {self._gpu_layers} GPU layers.")
#             await asyncio.wait_for(asyncio.to_thread(self._load_model_blocking), timeout=120)
#             test_response = await asyncio.to_thread(self.llm.create_chat_completion,messages=[{"role": "system", "content": "You are a concise assistant."},{"role": "user", "content": "Say hi."}],max_tokens=8,temperature=0.1)
#             if test_response and "choices" in test_response:
#                 self.is_initialized = True; self.model_info = {"model_path": self.model_path,"context_window": 2048,"threads": int(os.environ.get("OMP_NUM_THREADS", "4")),"gpu_layers": self._gpu_layers,"initialized_at": datetime.now().isoformat()}; logger.info("Local LLM service initialized successfully")
#             else:
#                 raise Exception("Model test failed")
#         except Exception as e:
#             logger.error(f"Error initializing local LLM service: {e}", exc_info=True); self.is_initialized = False

#     def _load_model_blocking(self):
#         from llama_cpp import Llama
#         logger.info(f"Loading model: {self.model_path} with n_gpu_layers={self._gpu_layers}")
#         self.llm = Llama(model_path=self.model_path,n_ctx=2048,n_threads=int(os.environ.get("OMP_NUM_THREADS", "4")),n_gpu_layers=self._gpu_layers,n_batch=128,f16_kv=True,logits_all=False,use_mmap=True,use_mlock=False,chat_format="llama-3",verbose=False)

#     def _probe_gpu_layers_supported(self, gpu_layers: int, timeout_sec: int = 45) -> bool:
#         if platform.system().lower() != "darwin" or gpu_layers <= 0: return False
#         ctx = mp.get_context("spawn"); proc = ctx.Process(target=_child_probe_for_metal, args=(self.model_path, gpu_layers)); proc.daemon = True; proc.start(); proc.join(timeout=timeout_sec)
#         if proc.is_alive():
#             logger.warning(f"Metal probe timed out."); proc.terminate(); proc.join(5); return False
#         return proc.exitcode == 0
    
#     def _check_model_file(self) -> bool: return os.path.exists(self.model_path)
    
#     async def generate_response(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
#         """Generate a response and parse it into the structure expected by main.py."""
#         try:
#             if not self.is_initialized: await self.initialize()
#             if not self.is_initialized: return self._get_fallback_response(query, language)
            
#             prompt = self._build_prompt(query, intent_result, weather_data, soil_data, rule_output, retrieved_docs, language)
            
#             # Using create_completion as it's robust and works well with the prompt format.
#             response = self.llm.create_completion(prompt, max_tokens=512, temperature=0.2, top_p=0.9, stop=["<|eot_id|>", "\n\n"], echo=False)
            
#             response_text = response["choices"][0]["text"].strip() if response and "choices" in response and len(response["choices"]) > 0 else "I apologize, I could not generate a response."
            
#             # CRITICAL FIX: Parse the response to get the full structure.
#             parsed_response = self._parse_response(response_text)

#             return {
#                 "response_text": response_text,
#                 "parsed_response": parsed_response,
#                 "model_info": self.model_info,
#                 "prompt_length": len(prompt),
#                 "generated_at": datetime.now().isoformat(),
#                 "language": language
#             }
#         except Exception as e:
#             logger.error(f"Error generating response with local LLM: {e}", exc_info=True)
#             return self._get_fallback_response(query, language)
    
#     def _build_prompt(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str) -> str:
#         intent = intent_result.get("intent", "default")
#         system_prompt = self.system_prompts.get(intent, self.system_prompts["default"])
#         prompt_template = self.prompt_templates.get(language, self.prompt_templates["en"])
#         context_info = self._format_context_info(weather_data, soil_data, rule_output, retrieved_docs)
#         prompt = prompt_template.format(system_prompt=system_prompt, context_info=context_info, user_query=query, language=language)
#         return prompt
    
#     def _format_context_info(self, weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]) -> str:
#         context_parts = []
#         if weather_data and weather_data.get('current'):
#             current = weather_data.get("current", {}); metrics = weather_data.get("irrigation_metrics", {})
#             context_parts.append(f"Weather: Temp {current.get('temperature', 'N/A')}°C, Humidity {current.get('humidity', 'N/A')}%, Rain Chance (24h) {metrics.get('rain_probability_24h', 'N/A')}%")
#         if soil_data and not soil_data.get('fallback'):
#             props = soil_data.get("soil_properties", {})
#             context_parts.append(f"Soil: Type {props.get('soil_type', 'N/A')}, pH {props.get('ph', 'N/A')}")
#         if rule_output:
#             context_parts.append(f"Rule Engine Decision: {rule_output.get('final_decision', 'N/A')} because {rule_output.get('primary_reason', 'N/A')}")
#         return "\n".join(context_parts) if context_parts else "No specific context data is available."

#     # CRITICAL FIX: Re-introducing the parsing logic.
#     def _parse_response(self, response_text: str) -> Dict[str, Any]:
#         """Parses the structured response from the LLM."""
#         answer, explanation, sources = response_text, "", []
#         try:
#             # Look for structured markers like "Answer:", "Explanation:"
#             answer_match = re.search(r"Answer:(.*?)(Explanation:|Sources:|$)", response_text, re.DOTALL | re.IGNORECASE)
#             explanation_match = re.search(r"Explanation:(.*?)(Sources:|$)", response_text, re.DOTALL | re.IGNORECASE)
#             sources_match = re.search(r"Sources:(.*)", response_text, re.DOTALL | re.IGNORECASE)

#             if answer_match:
#                 answer = answer_match.group(1).strip()
#             if explanation_match:
#                 explanation = explanation_match.group(1).strip()
#             if sources_match:
#                 sources = [s.strip() for s in sources_match.group(1).strip().split('\n') if s.strip()]

#             # If no markers found, return the whole text as the answer
#             if not answer_match and not explanation_match:
#                 return {"answer": response_text, "explanation": "", "sources": []}
            
#             return {"answer": answer, "explanation": explanation, "sources": sources}
#         except Exception:
#             return {"answer": response_text, "explanation": "", "sources": []}

#     def _get_fallback_response(self, query: str, language: str) -> Dict[str, Any]:
#         # ... (This function remains the same)
#         responses = {"hi": {"answer": "माफ़ करें, मैं अभी आपकी मदद नहीं कर सकता।", "explanation": "तकनीकी समस्या।"},"en": {"answer": "I apologize, but I'm unable to help right now.", "explanation": "Technical issue."}, "hinglish": {"answer": "Sorry, main abhi help nahi kar sakta.", "explanation": "Technical issue."}}
#         fallback = responses.get(language, responses["en"])
#         return {"response_text": fallback["answer"], "parsed_response": fallback, "model_info": {"status": "fallback"}, "prompt_length": 0, "generated_at": datetime.now().isoformat(), "language": language}

#     def _get_prompt_template(self) -> str:
#         """The robust prompt template using a simplified but effective format."""
#         return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
# **Context Data:**
# {context_info}

# **User's Question:** {user_query}

# Your task is to synthesize the data above to answer the user's question. Follow these instructions exactly:
# 1.  Provide a direct, one-sentence answer. Start with "Answer:".
# 2.  Provide a brief explanation for your answer, based ONLY on the context data. Start with "Explanation:".
# 3.  List the data sources you used. Start with "Sources:".
# Respond in {language}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

#     def _get_irrigation_system_prompt(self) -> str:
#         return "You are a data synthesis AI for agriculture. Your task is to analyze the provided data and directly answer the user's question. Do not use cautious language or disclaimers. If the data indicates a clear decision, state it confidently."

#     # --- (Other system prompts remain the same) ---
#     def _get_fertilizer_system_prompt(self) -> str: return "You are a data synthesis AI. Analyze the soil data to answer the user's fertilizer question."
#     def _get_pest_control_system_prompt(self) -> str: return "You are a data synthesis AI. Use the context to answer the user's pest control question."
#     def _get_weather_system_prompt(self) -> str: return "You are a data synthesis AI. Summarize the provided weather data to answer the user's question."
#     def _get_harvest_system_prompt(self) -> str: return "You are a data synthesis AI. Use the context to answer the user's harvest question."
#     def _get_default_system_prompt(self) -> str: return "You are a data synthesis AI. Use the provided context to answer the user's question directly."
    
#     def get_service_status(self) -> Dict[str, Any]:
#         return {"is_initialized": self.is_initialized, "model_path": self.model_path, "model_info": self.model_info}



# 2222222



# import logging
# import os
# import platform
# import asyncio
# import re
# from typing import Dict, Any, List, Optional
# from datetime import datetime
# import json
# import multiprocessing as mp

# # (The _child_probe_for_metal function remains the same)
# def _child_probe_for_metal(model_path: str, gpu_layers_arg: int):
#     try:
#         os.environ.pop("LLAMA_NO_METAL", None)
#         from llama_cpp import Llama
#         llm = Llama(model_path=model_path, n_ctx=1024, n_threads=2, n_gpu_layers=gpu_layers_arg, n_batch=64, f16_kv=True, logits_all=False, use_mmap=True, use_mlock=False, chat_format="llama-3", verbose=False)
#         _ = llm.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.1)
#         os._exit(0)
#     except Exception:
#         os._exit(1)

# from config import config

# logger = logging.getLogger(__name__)

# class LocalLLMService:
#     def __init__(self):
#         self.model_path = config.LOCAL_LLM_PATH
#         self.llm = None
#         self.is_initialized = False
#         self.model_info = {}
#         self._gpu_layers: int = 35
        
#         self.prompt_templates = {
#             "hi": self._get_prompt_template(),
#             "en": self._get_prompt_template(),
#             "hinglish": self._get_prompt_template()
#         }
        
#         self.system_prompts = {
#             "irrigation": self._get_irrigation_system_prompt(),
#             "fertilizer": self._get_fertilizer_system_prompt(),
#             "pest_control": self._get_pest_control_system_prompt(),
#             "weather": self._get_weather_system_prompt(),
#             "harvest": self._get_harvest_system_prompt(),
#             "default": self._get_default_system_prompt()
#         }
    
#     async def initialize(self):
#         # ... (This function is working perfectly, no changes needed) ...
#         try:
#             if platform.system().lower() == "darwin" and os.environ.get("ENABLE_LOCAL_LLM", "0") != "1":
#                 self.is_initialized = False; return
#             if platform.system().lower() == "darwin": os.environ.setdefault("OMP_NUM_THREADS", str(mp.cpu_count() // 2))
#             if not self._check_model_file(): self.is_initialized = False; return
#             gpu_safe = await asyncio.to_thread(self._probe_gpu_layers_supported, self._gpu_layers)
#             if not gpu_safe: self._gpu_layers = 0; os.environ["LLAMA_NO_METAL"] = "1"
#             await asyncio.wait_for(asyncio.to_thread(self._load_model_blocking), timeout=120)
#             test_response = await asyncio.to_thread(self.llm.create_chat_completion,messages=[{"role": "system", "content": "You are a concise assistant."},{"role": "user", "content": "Say hi."}],max_tokens=8,temperature=0.1)
#             if test_response and "choices" in test_response:
#                 self.is_initialized = True; self.model_info = {"model_path": self.model_path,"context_window": 2048,"threads": int(os.environ.get("OMP_NUM_THREADS", "4")),"gpu_layers": self._gpu_layers,"initialized_at": datetime.now().isoformat()}; logger.info("Local LLM service initialized successfully")
#             else: raise Exception("Model test failed")
#         except Exception as e:
#             logger.error(f"Error initializing local LLM service: {e}", exc_info=True); self.is_initialized = False

#     def _load_model_blocking(self):
#         from llama_cpp import Llama
#         self.llm = Llama(model_path=self.model_path,n_ctx=2048,n_threads=int(os.environ.get("OMP_NUM_THREADS", "4")),n_gpu_layers=self._gpu_layers,n_batch=128,f16_kv=True,logits_all=False,use_mmap=True,use_mlock=False,chat_format="llama-3",verbose=False)

#     def _probe_gpu_layers_supported(self, gpu_layers: int, timeout_sec: int = 45) -> bool:
#         if platform.system().lower() != "darwin" or gpu_layers <= 0: return False
#         ctx = mp.get_context("spawn"); proc = ctx.Process(target=_child_probe_for_metal, args=(self.model_path, gpu_layers)); proc.daemon = True; proc.start(); proc.join(timeout=timeout_sec)
#         if proc.is_alive(): proc.terminate(); proc.join(5); return False
#         return proc.exitcode == 0
    
#     def _check_model_file(self) -> bool: return os.path.exists(self.model_path)
    
#     async def generate_response(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
#         try:
#             if not self.is_initialized: await self.initialize()
#             if not self.is_initialized: return self._get_fallback_response(query, language)
            
#             prompt = self._build_prompt(query, intent_result, weather_data, soil_data, rule_output, retrieved_docs, language)
            
#             # --- QUALITY TUNING 1: Increase temperature for more detailed responses ---
#             response = self.llm.create_completion(prompt, max_tokens=512, temperature=0.4, top_p=0.9, stop=["<|eot_id|>", "User Query:"], echo=False)
            
#             response_text = response["choices"][0]["text"].strip() if response and "choices" in response and len(response["choices"]) > 0 else "I apologize, I could not generate a response."
            
#             parsed_response = self._parse_structured_response(response_text)

#             return { "response_text": response_text, "parsed_response": parsed_response, "model_info": self.model_info }
            
#         except Exception as e:
#             logger.error(f"Error generating response with local LLM: {e}", exc_info=True)
#             return self._get_fallback_response(query, language)
    
#     def _build_prompt(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str) -> str:
#         intent = intent_result.get("intent", "default")
#         system_prompt = self.system_prompts.get(intent, self.system_prompts["default"])
#         prompt_template = self.prompt_templates.get(language, self.prompt_templates["en"])
#         context_info = self._format_context_info(weather_data, soil_data, rule_output, retrieved_docs)
#         return prompt_template.format(system_prompt=system_prompt, context_info=context_info, user_query=query, language=language)
    
#     # --- QUALITY TUNING 2: Make the context more descriptive for the LLM ---
#     def _format_context_info(self, weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]) -> str:
#         context_parts = []
#         if weather_data and weather_data.get('current'):
#             current = weather_data.get("current", {}); metrics = weather_data.get("irrigation_metrics", {})
#             context_parts.append(f"- Weather: Temp is {current.get('temperature', 'N/A')}°C, Humidity is {current.get('humidity', 'N/A')}%, and the 24-hour rain chance is {metrics.get('rain_probability_24h', 'N/A')}%")
        
#         if soil_data and not soil_data.get('fallback'):
#             props = soil_data.get("soil_properties", {}); loc = soil_data.get("location", {})
#             # This is the key change: tell the AI how far the data is from!
#             distance_info = f"(from a data point {loc.get('distance_km', 'N/A'):.1f} km away)"
#             context_parts.append(f"- Soil {distance_info}: The estimated soil type is {props.get('soil_type', 'N/A')} with a pH of {props.get('ph', 'N/A')}.")

#         if rule_output:
#             reason = rule_output.get('primary_reason', 'N/A')
#             moisture = rule_output.get('estimated_soil_moisture', 'N/A')
#             context_parts.append(f"- Rule Engine Analysis: The preliminary decision is to '{rule_output.get('final_decision', 'N/A')}' because the estimated soil moisture is {moisture}%. Reason: {reason}")
        
#         if retrieved_docs:
#             context_parts.append(f"- Relevant Knowledge: {retrieved_docs[0].get('content', 'N/A')[:250]}...")
            
#         return "\n".join(context_parts) if context_parts else "No specific context data is available for this query."

#     def _parse_structured_response(self, text: str) -> Dict[str, Any]:
#         """A more robust parser for the new, detailed response format."""
#         try:
#             recommendation = re.search(r"\*\*Recommendation:\*\*(.*?)\*\*Reasoning:\*\*", text, re.DOTALL | re.IGNORECASE)
#             reasoning = re.search(r"\*\*Reasoning:\*\*(.*?)\*\*Data Points:\*\*", text, re.DOTALL | re.IGNORECASE)
            
#             answer = recommendation.group(1).strip() if recommendation else text
#             explanation = reasoning.group(1).strip() if reasoning else ""

#             return {"answer": answer, "explanation": explanation, "sources": []}
#         except Exception:
#             return {"answer": text, "explanation": "", "sources": []}

#     def _get_fallback_response(self, query: str, language: str) -> Dict[str, Any]:
#         # ... (This function remains the same) ...
#         responses = {"hi": {},"en": {"answer": "I apologize, but I'm unable to help right now.", "explanation": "Technical issue."}}
#         fallback = responses.get(language, responses["en"])
#         return {"response_text": fallback.get("answer"), "parsed_response": fallback, "model_info": {"status": "fallback"}}

#     # --- QUALITY TUNING 3: A more sophisticated prompt that encourages detail ---
#     def _get_prompt_template(self) -> str:
#         """A new, high-quality prompt for generating detailed answers."""
#         return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
# **Context Data:**
# {context_info}

# **User's Question:** {user_query}

# **Your Task:**
# Based on all the context data provided, synthesize a comprehensive and practical recommendation for the user. Follow this structure precisely:
# 1.  **Recommendation:** Start with a clear, direct answer to the user's question.
# 2.  **Reasoning:** Explain *why* you are making this recommendation, referencing specific data points from the context (e.g., "Because the rain chance is low and the estimated moisture is adequate..."). If any data is of low quality (like soil data being far away), mention this as a caveat in your reasoning.
# 3.  **Data Points:** Briefly list the most critical data points that led to your decision.

# Respond in {language}.
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# **Recommendation:** """ # We pre-fill the start of the response to guide the model

#     def _get_irrigation_system_prompt(self) -> str:
#         return """You are an expert agricultural AI advisor. Your task is to synthesize the provided data into a clear, actionable irrigation plan. Do not refuse to answer. Use all the data to form a conclusion."""

#     # ... (Other system prompts remain the same) ...
#     def _get_fertilizer_system_prompt(self) -> str: return "You are an expert agricultural AI advisor for fertilizers."
#     def _get_pest_control_system_prompt(self) -> str: return "You are an expert agricultural AI advisor for pest control."
#     def _get_weather_system_prompt(self) -> str: return "You are an expert agricultural AI advisor for weather."
#     def _get_harvest_system_prompt(self) -> str: return "You are an expert agricultural AI advisor for harvesting."
#     def _get_default_system_prompt(self) -> str: return "You are a helpful agricultural AI advisor."
    
#     def get_service_status(self) -> Dict[str, Any]:
#         return {"is_initialized": self.is_initialized, "model_path": self.model_path, "model_info": self.model_info}




#33333

# in app/modules/llm_local.py

# import logging
# import os
# import platform
# import asyncio
# import re
# from typing import Dict, Any, List
# from datetime import datetime
# import json
# import multiprocessing as mp

# # (The _child_probe_for_metal function remains the same)
# def _child_probe_for_metal(model_path: str, gpu_layers_arg: int):
#     try:
#         os.environ.pop("LLAMA_NO_METAL", None); from llama_cpp import Llama
#         llm = Llama(model_path=model_path, n_ctx=1024, n_threads=2, n_gpu_layers=gpu_layers_arg, n_batch=64, f16_kv=True, logits_all=False, use_mmap=True, use_mlock=False, chat_format="llama-3", verbose=False)
#         _ = llm.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.1)
#         os._exit(0)
#     except Exception: os._exit(1)

# from config import config

# logger = logging.getLogger(__name__)

# class LocalLLMService:
#     def __init__(self):
#         self.model_path = config.LOCAL_LLM_PATH; self.llm = None; self.is_initialized = False; self.model_info = {}; self._gpu_layers: int = 35
#         self.prompt_templates = {"hi": self._get_prompt_template(), "en": self._get_prompt_template(), "hinglish": self._get_prompt_template()}
#         self.system_prompts = {"irrigation": self._get_irrigation_system_prompt(), "fertilizer": self._get_fertilizer_system_prompt(), "pest_control": self._get_pest_control_system_prompt(), "weather": self._get_weather_system_prompt(), "harvest": self._get_harvest_system_prompt(), "default": self._get_default_system_prompt()}
    
#     async def initialize(self):
#         try:
#             if platform.system().lower() == "darwin" and os.environ.get("ENABLE_LOCAL_LLM", "0") != "1": self.is_initialized = False; return
#             if platform.system().lower() == "darwin": os.environ.setdefault("OMP_NUM_THREADS", str(mp.cpu_count() // 2))
#             if not self._check_model_file(): logger.error(f"Model file not found: {self.model_path}"); self.is_initialized = False; return
#             gpu_safe = await asyncio.to_thread(self._probe_gpu_layers_supported, self._gpu_layers)
#             if not gpu_safe: self._gpu_layers = 0; os.environ["LLAMA_NO_METAL"] = "1"
#             await asyncio.wait_for(asyncio.to_thread(self._load_model_blocking), timeout=120)
#             test_response = await asyncio.to_thread(self.llm.create_chat_completion,messages=[{"role": "system", "content": "You are a concise assistant."},{"role": "user", "content": "Say hi."}],max_tokens=8,temperature=0.1)
#             if test_response and "choices" in test_response:
#                 self.is_initialized = True; self.model_info = {"model_path": self.model_path,"context_window": 2048,"threads": int(os.environ.get("OMP_NUM_THREADS", "4")),"gpu_layers": self._gpu_layers,"initialized_at": datetime.now().isoformat()}; logger.info("Local LLM service initialized successfully")
#             else: raise Exception("Model test failed")
#         except Exception as e:
#             logger.error(f"Error initializing local LLM service: {e}", exc_info=True); self.is_initialized = False

#     def _load_model_blocking(self):
#         from llama_cpp import Llama
#         self.llm = Llama(model_path=self.model_path,n_ctx=2048,n_threads=int(os.environ.get("OMP_NUM_THREADS", "4")),n_gpu_layers=self._gpu_layers,n_batch=128,f16_kv=True,logits_all=False,use_mmap=True,use_mlock=False,chat_format="llama-3",verbose=False)

#     def _probe_gpu_layers_supported(self, gpu_layers: int, timeout_sec: int = 45) -> bool:
#         if platform.system().lower() != "darwin" or gpu_layers <= 0: return False
#         ctx = mp.get_context("spawn"); proc = ctx.Process(target=_child_probe_for_metal, args=(self.model_path, gpu_layers)); proc.daemon = True; proc.start(); proc.join(timeout=timeout_sec)
#         if proc.is_alive(): proc.terminate(); proc.join(5); return False
#         return proc.exitcode == 0
    
#     def _check_model_file(self) -> bool: return os.path.exists(self.model_path)
    
#     async def generate_response(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
#         try:
#             if not self.is_initialized: await self.initialize()
#             if not self.is_initialized: return self._get_fallback_response(query, language)
            
#             prompt = self._build_prompt(query, intent_result, weather_data, soil_data, rule_output, retrieved_docs, language)
            
#             # Increase temperature for more creative, chatbot-like answers
#             response = self.llm.create_completion(prompt, max_tokens=512, temperature=0.5, top_p=0.9, stop=["<|eot_id|>", "User Query:"], echo=False)
            
#             response_text = response["choices"][0]["text"].strip() if response and "choices" in response and len(response["choices"]) > 0 else "I apologize, I could not generate a response."
            
#             parsed_response = self._parse_response(response_text)

#             return { "response_text": response_text, "parsed_response": parsed_response, "model_info": self.model_info }
#         except Exception as e:
#             logger.error(f"Error generating response with local LLM: {e}", exc_info=True)
#             return self._get_fallback_response(query, language)
    
#     def _build_prompt(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str) -> str:
#         intent = intent_result.get("intent", "default")
#         system_prompt = self.system_prompts.get(intent, self.system_prompts["default"])
#         prompt_template = self.prompt_templates.get(language, self.prompt_templates["en"])
#         context_info = self._format_context_info(weather_data, soil_data, rule_output)
#         return prompt_template.format(system_prompt=system_prompt, context_info=context_info, user_query=query, language=language)
    
#     def _format_context_info(self, weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any]) -> str:
#         context_parts = []
#         if weather_data and weather_data.get('current'):
#             current = weather_data.get("current", {}); metrics = weather_data.get("irrigation_metrics", {})
#             context_parts.append(f"- Weather: The current temperature is {current.get('temperature', 'N/A')}°C with {current.get('humidity', 'N/A')}% humidity. The 24-hour rain chance is {metrics.get('rain_probability_24h', 'N/A')}%.")
#         if soil_data and not soil_data.get('fallback'):
#             props = soil_data.get("soil_properties", {}); loc = soil_data.get("location", {})
#             distance_info = f"(Note: this data is from a point {loc.get('distance_km', 'N/A'):.1f} km away)"
#             context_parts.append(f"- Soil: The estimated soil type is {props.get('soil_type', 'N/A')}. {distance_info}")
#         if rule_output:
#             reason = rule_output.get('primary_reason', 'N/A')
#             context_parts.append(f"- Current Situation: The rule engine recommends you '{rule_output.get('final_decision', 'N/A')}' because: {reason}")
#         return "\n".join(context_parts) if context_parts else "No specific context data is available for this query."

#     def _parse_response(self, text: str) -> Dict[str, Any]:
#         """A simple but robust parser to separate the main point from the details."""
#         parts = text.split('\n\n', 1)
#         answer = parts[0].strip()
#         explanation = parts[1].strip() if len(parts) > 1 else ""
#         return {"answer": answer, "explanation": explanation, "sources": []}

#     def _get_fallback_response(self, query: str, language: str) -> Dict[str, Any]:
#         responses = {"en": {"answer": "I apologize, an error occurred.", "explanation": "Could not generate a response due to a technical issue."}}
#         fallback = responses.get(language, responses["en"])
#         return {"response_text": fallback.get("answer"), "parsed_response": fallback, "model_info": {"status": "fallback"}}

#     # --- FINAL QUALITY UPGRADE 1: THE PROMPT TEMPLATE ---
#     def _get_prompt_template(self) -> str:
#         """A sophisticated prompt that encourages both specific and general advice."""
#         return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
# **Context Data:**
# {context_info}

# **User's Question:** {user_query}

# **Your Task:**
# Act as a friendly, expert agricultural chatbot. Synthesize the context data and your own knowledge to provide a helpful, two-part response.
# 1.  **Immediate Recommendation:** First, directly answer the user's question based on the "Current Situation" from the context data.
# 2.  **General Advice:** In a new paragraph, provide a helpful "General Best Practice" that addresses the broader topic of the user's question (e.g., if they ask about irrigating today, give general advice on the best time of day to irrigate).

# Respond in {language}.
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

#     # --- FINAL QUALITY UPGRADE 2: THE SYSTEM PROMPT ---
#     def _get_irrigation_system_prompt(self) -> str:
#         return "You are 'Agri-AI', an expert agricultural advisor. Your goal is to give helpful, data-driven irrigation advice to farmers in a clear and conversational way."

#     # --- (Other system prompts can be updated similarly) ---
#     def _get_fertilizer_system_prompt(self) -> str: return "You are 'Agri-AI', an expert agricultural advisor specializing in soil health and fertilizers."
#     def _get_pest_control_system_prompt(self) -> str: return "You are 'Agri-AI', an expert advisor for integrated pest management."
#     def _get_weather_system_prompt(self) -> str: return "You are 'Agri-AI', an expert at interpreting weather data for farmers."
#     def _get_harvest_system_prompt(self) -> str: return "You are 'Agri-AI', an expert advisor on optimal harvesting techniques."
#     def _get_default_system_prompt(self) -> str: return "You are 'Agri-AI', a helpful and friendly agricultural advisor."
    
#     def get_service_status(self) -> Dict[str, Any]:
#         return {"is_initialized": self.is_initialized, "model_path": self.model_path, "model_info": self.model_info}





#44444# in app/modules/llm_local.py

import logging
import os
import platform
import asyncio
import re
from typing import Dict, Any, List
from datetime import datetime
import json
import multiprocessing as mp

# (The _child_probe_for_metal function remains the same)
def _child_probe_for_metal(model_path: str, gpu_layers_arg: int):
    try:
        os.environ.pop("LLAMA_NO_METAL", None); from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=1024, n_threads=2, n_gpu_layers=gpu_layers_arg, n_batch=64, f16_kv=True, logits_all=False, use_mmap=True, use_mlock=False, chat_format="llama-3", verbose=False)
        _ = llm.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.1)
        os._exit(0)
    except Exception: os._exit(1)

from config import config

logger = logging.getLogger(__name__)

class LocalLLMService:
    def __init__(self):
        self.model_path = config.LOCAL_LLM_PATH; self.llm = None; self.is_initialized = False; self.model_info = {}; self._gpu_layers: int = 35
        self.prompt_templates = {"hi": self._get_prompt_template(), "en": self._get_prompt_template(), "hinglish": self._get_prompt_template()}
        self.system_prompts = {"irrigation": self._get_irrigation_system_prompt(), "fertilizer": self._get_fertilizer_system_prompt(), "pest_control": self._get_pest_control_system_prompt(), "weather": self._get_weather_system_prompt(), "harvest": self._get_harvest_system_prompt(), "default": self._get_default_system_prompt()}
    
    async def initialize(self):
        # ... (This function is working perfectly, no changes needed) ...
        try:
            if platform.system().lower() == "darwin" and os.environ.get("ENABLE_LOCAL_LLM", "0") != "1": self.is_initialized = False; return
            if platform.system().lower() == "darwin": os.environ.setdefault("OMP_NUM_THREADS", str(mp.cpu_count() // 2))
            if not self._check_model_file(): logger.error(f"Model file not found: {self.model_path}"); self.is_initialized = False; return
            gpu_safe = await asyncio.to_thread(self._probe_gpu_layers_supported, self._gpu_layers)
            if not gpu_safe: self._gpu_layers = 0; os.environ["LLAMA_NO_METAL"] = "1"
            await asyncio.wait_for(asyncio.to_thread(self._load_model_blocking), timeout=120)
            test_response = await asyncio.to_thread(self.llm.create_chat_completion,messages=[{"role": "system", "content": "You are a concise assistant."},{"role": "user", "content": "Say hi."}],max_tokens=8,temperature=0.1)
            if test_response and "choices" in test_response:
                self.is_initialized = True; self.model_info = {"model_path": self.model_path,"context_window": 2048,"threads": int(os.environ.get("OMP_NUM_THREADS", "4")),"gpu_layers": self._gpu_layers,"initialized_at": datetime.now().isoformat()}; logger.info("Local LLM service initialized successfully")
            else: raise Exception("Model test failed")
        except Exception as e:
            logger.error(f"Error initializing local LLM service: {e}", exc_info=True); self.is_initialized = False

    def _load_model_blocking(self):
        from llama_cpp import Llama
        self.llm = Llama(model_path=self.model_path,n_ctx=2048,n_threads=int(os.environ.get("OMP_NUM_THREADS", "4")),n_gpu_layers=self._gpu_layers,n_batch=128,f16_kv=True,logits_all=False,use_mmap=True,use_mlock=False,chat_format="llama-3",verbose=False)

    def _probe_gpu_layers_supported(self, gpu_layers: int, timeout_sec: int = 45) -> bool:
        if platform.system().lower() != "darwin" or gpu_layers <= 0: return False
        ctx = mp.get_context("spawn"); proc = ctx.Process(target=_child_probe_for_metal, args=(self.model_path, gpu_layers)); proc.daemon = True; proc.start(); proc.join(timeout=timeout_sec)
        if proc.is_alive(): proc.terminate(); proc.join(5); return False
        return proc.exitcode == 0
    
    def _check_model_file(self) -> bool: return os.path.exists(self.model_path)
    
    async def generate_response(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
        try:
            if not self.is_initialized: await self.initialize()
            if not self.is_initialized: return self._get_fallback_response(query, language)
            prompt = self._build_prompt(query, intent_result, weather_data, soil_data, rule_output, retrieved_docs, language)
            response = self.llm.create_completion(prompt, max_tokens=512, temperature=0.5, top_p=0.9, stop=["<|eot_id|>", "User Query:"], echo=False)
            response_text = response["choices"][0]["text"].strip() if response and "choices" in response and len(response["choices"]) > 0 else "I apologize, I could not generate a response."
            parsed_response = self._parse_response(response_text)
            return { "response_text": response_text, "parsed_response": parsed_response, "model_info": self.model_info }
        except Exception as e:
            logger.error(f"Error generating response with local LLM: {e}", exc_info=True)
            return self._get_fallback_response(query, language)
    
    def _build_prompt(self, query: str, intent_result: Dict[str, Any], weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], language: str) -> str:
        intent = intent_result.get("intent", "default")
        system_prompt = self.system_prompts.get(intent, self.system_prompts.get(intent, self.system_prompts["default"]))
        prompt_template = self.prompt_templates.get(language, self.prompt_templates["en"])
        context_info = self._format_context_info(weather_data, soil_data, rule_output, retrieved_docs)
        return prompt_template.format(system_prompt=system_prompt, context_info=context_info, user_query=query, language=language)
    
    # --- THIS IS THE FINAL FIX ---
    def _format_context_info(self, weather_data: Dict[str, Any], soil_data: Dict[str, Any], rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        if weather_data and weather_data.get('current'):
            current = weather_data.get("current", {}); metrics = weather_data.get("irrigation_metrics", {})
            context_parts.append(f"- Weather: The current temperature is {current.get('temperature', 'N/A')}°C with {current.get('humidity', 'N/A')}% humidity. The 24-hour rain chance is {metrics.get('rain_probability_24h', 'N/A')}%.")
        
        if soil_data and not soil_data.get('fallback'):
            props = soil_data.get("soil_properties", {}); loc = soil_data.get("location", {})
            distance_info = f"(Note: this data is from a point {loc.get('distance_km', 'N/A'):.1f} km away)"
            
            # THE CRITICAL CHANGE IS HERE: We now include the NPK values in the string.
            soil_type = props.get('soil_type', 'N/A')
            ph = props.get('ph', 'N/A')
            n = props.get('nitrogen_kg_ha', 'N/A')
            p = props.get('phosphorus_kg_ha', 'N/A')
            k = props.get('potassium_kg_ha', 'N/A')
            context_parts.append(f"- Soil {distance_info}: Type is {soil_type}, pH {ph}. Nutrient levels (N-P-K) are {n}-{p}-{k} kg/ha.")

        if rule_output:
            reason = rule_output.get('primary_reason', 'N/A')
            context_parts.append(f"- Current Situation: The rule engine recommends you '{rule_output.get('final_decision', 'N/A')}' because: {reason}")
            
        return "\n".join(context_parts) if context_parts else "No specific context data is available for this query."

    def _parse_response(self, text: str) -> Dict[str, Any]:
        parts = text.split('\n\n', 1)
        answer = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else ""
        return {"answer": answer, "explanation": explanation, "sources": []}

    def _get_fallback_response(self, query: str, language: str) -> Dict[str, Any]:
        responses = {"en": {"answer": "I apologize, an error occurred.", "explanation": "Could not generate a response due to a technical issue."}}
        fallback = responses.get(language, responses["en"])
        return {"response_text": fallback.get("answer"), "parsed_response": fallback, "model_info": {"status": "fallback"}}

    def _get_prompt_template(self) -> str:
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
**Context Data:**
{context_info}

**User's Question:** {user_query}

**Your Task:**
Act as a friendly, expert agricultural chatbot. Synthesize the context data and your own knowledge to provide a helpful, two-part response.
1.  **Immediate Recommendation:** First, directly answer the user's question based on the provided "Context Data".
2.  **General Advice:** In a new paragraph, provide a helpful "General Best Practice" that addresses the broader topic of the user's question.

Respond in {language}.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def _get_irrigation_system_prompt(self) -> str: return "You are 'Agri-AI', an expert agricultural advisor. Your goal is to give helpful, data-driven irrigation advice to farmers in a clear and conversational way."
    def _get_fertilizer_system_prompt(self) -> str: return "You are 'Agri-AI', an expert agricultural advisor specializing in soil health and fertilizers."
    # ... (other prompts)
    def _get_pest_control_system_prompt(self) -> str: return "You are 'Agri-AI', an expert advisor for integrated pest management."
    def _get_weather_system_prompt(self) -> str: return "You are 'Agri-AI', an expert at interpreting weather data for farmers."
    def _get_harvest_system_prompt(self) -> str: return "You are 'Agri-AI', an expert advisor on optimal harvesting techniques."
    def _get_default_system_prompt(self) -> str: return "You are 'Agri-AI', a helpful and friendly agricultural advisor."
    
    def get_service_status(self) -> Dict[str, Any]:
        return {"is_initialized": self.is_initialized, "model_path": self.model_path, "model_info": self.model_info}


