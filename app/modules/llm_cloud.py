import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from config import config

logger = logging.getLogger(__name__)

class CloudLLMService:
    """Cloud LLM service using OpenAI GPT-4 as fallback"""
    
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.client = None
        self.is_available = bool(self.api_key)
        self.model_name = "gpt-4"  # Can be changed to gpt-3.5-turbo for cost optimization
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
    
    async def initialize(self):
        """Initialize the cloud LLM service"""
        try:
            if not self.is_available:
                logger.warning("Cloud LLM service not available - no API key")
                return
            
            # Import OpenAI client
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            
            # Test the connection
            test_response = await self._test_connection()
            if test_response:
                logger.info("Cloud LLM service initialized successfully")
            else:
                self.is_available = False
                logger.error("Failed to initialize cloud LLM service")
                
        except ImportError:
            logger.error("OpenAI Python client not installed")
            self.is_available = False
        except Exception as e:
            logger.error(f"Error initializing cloud LLM service: {e}")
            self.is_available = False
    
    async def _test_connection(self) -> bool:
        """Test the OpenAI API connection"""
        try:
            if not self.client:
                return False
            
            # Simple test call
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for testing
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return response is not None
            
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False
    
    async def generate_response(self, query: str, intent_result: Dict[str, Any], 
                              weather_data: Dict[str, Any], soil_data: Dict[str, Any],
                              rule_output: Dict[str, Any], retrieved_docs: List[Dict[str, Any]],
                              language: str = "en") -> Dict[str, Any]:
        """
        Generate response using OpenAI GPT-4
        
        Args:
            query: User query
            intent_result: Intent and entity extraction result
            weather_data: Weather information
            soil_data: Soil information
            rule_output: Rule engine output
            retrieved_docs: Retrieved relevant documents
            language: Response language
            
        Returns:
            Generated response dictionary
        """
        try:
            if not self.is_available:
                return self._get_fallback_response(query, language)
            
            if not self.client:
                await self.initialize()
            
            if not self.client:
                return self._get_fallback_response(query, language)
            
            # Build the prompt
            system_prompt = self._build_system_prompt(intent_result, language)
            user_prompt = self._build_user_prompt(
                query, weather_data, soil_data, rule_output, retrieved_docs, language
            )
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract response text
            if response and response.choices:
                response_text = response.choices[0].message.content.strip()
            else:
                response_text = "I apologize, but I couldn't generate a proper response."
            
            # Parse the response
            parsed_response = self._parse_response(response_text, language)
            
            return {
                "response_text": response_text,
                "parsed_response": parsed_response,
                "model_info": {
                    "model": self.model_name,
                    "provider": "OpenAI",
                    "generated_at": datetime.now().isoformat()
                },
                "prompt_length": len(system_prompt) + len(user_prompt),
                "language": language,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response with cloud LLM: {e}")
            return self._get_fallback_response(query, language)
    
    def _build_system_prompt(self, intent_result: Dict[str, Any], language: str) -> str:
        """Build the system prompt for the cloud LLM"""
        try:
            intent = intent_result.get("intent", "default")
            
            # Base system prompt
            base_prompt = """You are an expert agricultural advisor specializing in {intent}. Your role is to provide accurate, practical advice based on:

1. Current weather conditions and forecasts
2. Soil moisture levels and properties  
3. Crop-specific requirements
4. Rule-based analysis results
5. Relevant agricultural knowledge from the knowledge base

IMPORTANT REQUIREMENTS:
- Always base your recommendations on the provided data
- Be specific about timing, amounts, and methods
- Prioritize water conservation and sustainable practices
- If data is insufficient, say so and suggest what additional information is needed
- Provide clear, actionable advice that farmers can implement immediately
- Always cite your sources and explain your reasoning

Response Format:
1. Answer: Provide a clear, concise answer to the user's question
2. Explanation: Explain the reasoning behind your recommendation
3. Sources: List the sources of information used (weather data, soil data, knowledge base documents)

Language: Respond in {language} language."""
            
            # Intent-specific additions
            intent_specific = self._get_intent_specific_prompt(intent)
            
            return base_prompt.format(intent=intent, language=language) + "\n\n" + intent_specific
            
        except Exception as e:
            logger.error(f"Error building system prompt: {e}")
            return "You are an expert agricultural advisor. Provide helpful, accurate advice based on the information provided."
    
    def _get_intent_specific_prompt(self, intent: str) -> str:
        """Get intent-specific prompt additions"""
        intent_prompts = {
            "irrigation": """
IRRIGATION-SPECIFIC RULES:
- If rain is forecasted (>60% probability), recommend skipping irrigation unless soil moisture is critically low
- Consider soil type: sandy soils need frequent light irrigation, clay soils need less frequent deep irrigation
- Factor in crop growth stage and water requirements
- Recommend optimal irrigation timing (early morning or evening to reduce evaporation)
- Suggest appropriate irrigation methods (drip, sprinkler, flood) based on conditions""",
            
            "fertilizer": """
FERTILIZER-SPECIFIC RULES:
- Base recommendations on soil test results and crop requirements
- Consider soil pH and nutrient availability
- Recommend appropriate timing and application methods
- Suggest organic alternatives when possible
- Consider cost-effectiveness and environmental impact""",
            
            "pest_control": """
PEST CONTROL-SPECIFIC RULES:
- Recommend integrated pest management (IPM) approaches
- Consider beneficial insects and natural predators
- Suggest least-toxic chemical options when necessary
- Recommend proper timing for maximum effectiveness
- Consider crop rotation and cultural practices""",
            
            "weather": """
WEATHER-SPECIFIC RULES:
- Interpret weather data in farming context
- Consider seasonal patterns and crop requirements
- Recommend protective measures for extreme weather
- Suggest optimal timing for weather-sensitive operations
- Consider long-term climate trends""",
            
            "harvest": """
HARVEST-SPECIFIC RULES:
- Base timing on crop maturity indicators
- Consider weather conditions for optimal harvest
- Recommend proper harvesting methods and equipment
- Suggest post-harvest handling and storage
- Consider market timing and quality requirements"""
        }
        
        return intent_prompts.get(intent, "")
    
    def _build_user_prompt(self, query: str, weather_data: Dict[str, Any], 
                          soil_data: Dict[str, Any], rule_output: Dict[str, Any],
                          retrieved_docs: List[Dict[str, Any]], language: str) -> str:
        """Build the user prompt with context information"""
        try:
            context_parts = []
            
            # Weather information
            if weather_data:
                current = weather_data.get("current", {})
                irrigation_metrics = weather_data.get("irrigation_metrics", {})
                
                weather_info = f"""
WEATHER INFORMATION:
- Temperature: {current.get('temperature', 'N/A')}°C
- Humidity: {current.get('humidity', 'N/A')}%
- Rain probability (24h): {irrigation_metrics.get('rain_probability_24h', 'N/A')}%
- Rain probability (48h): {irrigation_metrics.get('rain_probability_48h', 'N/A')}%
- Total rain (7 days): {irrigation_metrics.get('total_rain_7d', 'N/A')} mm
- Evapotranspiration: {irrigation_metrics.get('evapotranspiration_daily', 'N/A')} mm/day
- Risk factors: {', '.join(irrigation_metrics.get('risk_factors', []))}
"""
                context_parts.append(weather_info)
            
            # Soil information
            if soil_data:
                current_status = soil_data.get("current_status", {})
                soil_properties = soil_data.get("soil_properties", {})
                irrigation_analysis = soil_data.get("irrigation_analysis", {})
                
                soil_info = f"""
SOIL INFORMATION:
- Soil type: {soil_properties.get('soil_type', 'N/A')}
- Current moisture: {current_status.get('moisture_percentage', 'N/A')}%
- Moisture status: {current_status.get('moisture_status', 'N/A')}
- Field capacity: {soil_properties.get('field_capacity', 'N/A')}%
- Wilting point: {soil_properties.get('wilting_point', 'N/A')}%
- Moisture deficit: {irrigation_analysis.get('moisture_deficit', 'N/A')}%
- Irrigation depth needed: {irrigation_analysis.get('irrigation_depth_mm', 'N/A')} mm
- pH: {soil_properties.get('ph', 'N/A')}
- Organic carbon: {soil_properties.get('organic_carbon', 'N/A')}%
"""
                context_parts.append(soil_info)
            
            # Rule engine output
            if rule_output:
                rule_info = f"""
RULE ENGINE ANALYSIS:
- Decision: {rule_output.get('final_decision', 'N/A')}
- Reason: {rule_output.get('primary_reason', 'N/A')}
- Priority: {rule_output.get('priority', 'N/A')}
- Confidence: {rule_output.get('confidence', 'N/A')}
- Rules applied: {', '.join(rule_output.get('rules_applied', []))}
"""
                context_parts.append(rule_info)
            
            # Retrieved documents
            if retrieved_docs:
                docs_info = "RELEVANT KNOWLEDGE BASE DOCUMENTS:\n"
                for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 documents
                    docs_info += f"""
Document {i}:
- Source: {doc.get('filename', 'N/A')}
- Content: {doc.get('content', 'N/A')[:300]}...
- Relevance Score: {doc.get('similarity_score', 'N/A'):.3f}
"""
                context_parts.append(docs_info)
            
            # Build final prompt
            context = "\n".join(context_parts)
            
            return f"""Based on the following information, please answer the user's question:

{context}

USER QUESTION: {query}

Please provide a comprehensive response in {language} language following the format specified in the system prompt."""
            
        except Exception as e:
            logger.error(f"Error building user prompt: {e}")
            return f"Please answer this question in {language}: {query}"
    
    def _parse_response(self, response_text: str, language: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format"""
        try:
            # Simple parsing - can be enhanced
            lines = response_text.split('\n')
            
            # Extract key information
            answer = ""
            explanation = ""
            sources = []
            
            current_section = "answer"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers in different languages
                if any(header in line.lower() for header in ["explanation:", "व्याख्या:", "explanation", "व्याख्या"]):
                    current_section = "explanation"
                    continue
                elif any(header in line.lower() for header in ["sources:", "स्रोत:", "sources", "स्रोत"]):
                    current_section = "sources"
                    continue
                elif any(header in line.lower() for header in ["answer:", "जवाब:", "answer", "जवाब"]):
                    current_section = "answer"
                    continue
                
                if current_section == "answer":
                    answer += line + " "
                elif current_section == "explanation":
                    explanation += line + " "
                elif current_section == "sources":
                    sources.append(line)
            
            # If no clear sections found, treat entire response as answer
            if not answer.strip():
                answer = response_text
            
            return {
                "answer": answer.strip(),
                "explanation": explanation.strip(),
                "sources": sources,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "answer": response_text,
                "explanation": "",
                "sources": [],
                "language": language
            }
    
    def get_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence score for the response"""
        try:
            # Base confidence for cloud LLM
            base_confidence = 0.8
            
            # Adjust based on response quality
            response_text = response.get("response_text", "")
            if len(response_text) > 150:
                base_confidence += 0.1
            
            # Check for structured response
            parsed = response.get("parsed_response", {})
            if parsed.get("answer") and parsed.get("explanation"):
                base_confidence += 0.1
            
            # Cap at 0.95
            return min(0.95, base_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.8
    
    def _get_fallback_response(self, query: str, language: str) -> Dict[str, Any]:
        """Get fallback response when cloud LLM fails"""
        fallback_responses = {
            "hi": {
                "answer": "माफ़ करें, मैं अभी आपकी मदद नहीं कर पा रहा हूं। कृपया थोड़ी देर बाद कोशिश करें।",
                "explanation": "तकनीकी समस्या के कारण प्रतिक्रिया उत्पन्न नहीं हो पा रही है।",
                "sources": []
            },
            "en": {
                "answer": "I apologize, but I'm unable to help you right now. Please try again later.",
                "explanation": "Unable to generate response due to technical issues.",
                "sources": []
            },
            "hinglish": {
                "answer": "Sorry, main abhi aapki help nahi kar pa raha hun. Please thodi der baad try karein.",
                "explanation": "Technical issue ke karan response generate nahi ho pa raha hai.",
                "sources": []
            }
        }
        
        fallback = fallback_responses.get(language, fallback_responses["en"])
        
        return {
            "response_text": fallback["answer"],
            "parsed_response": fallback,
            "model_info": {"status": "fallback", "provider": "OpenAI"},
            "prompt_length": 0,
            "language": language,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information"""
        return {
            "is_available": self.is_available,
            "provider": "OpenAI",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
            "client_initialized": self.client is not None
        }
    
    def update_model(self, model_name: str):
        """Update the model being used"""
        try:
            if model_name in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]:
                self.model_name = model_name
                logger.info(f"Updated model to {model_name}")
            else:
                logger.warning(f"Invalid model name: {model_name}")
                
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    def get_cost_estimate(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        """Get cost estimate for the API call"""
        try:
            # Approximate costs per 1K tokens (as of 2024)
            costs = {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
            }
            
            model_costs = costs.get(self.model_name, costs["gpt-4"])
            
            input_cost = (prompt_tokens / 1000) * model_costs["input"]
            output_cost = (completion_tokens / 1000) * model_costs["output"]
            total_cost = input_cost + output_cost
            
            return {
                "model": self.model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
                "total_cost_usd": round(total_cost, 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost estimate: {e}")
            return {"error": "Unable to calculate cost"}
