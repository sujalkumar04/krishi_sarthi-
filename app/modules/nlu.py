# in app/modules/nlu.py

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IntentExtractor:
    """
    A robust, keyword-based intent and entity extractor that reliably handles
    English, Hindi, and Hinglish queries for agricultural topics.
    """

    def __init__(self):
        # --- A comprehensive list of keywords for each intent ---
        # We combine all languages here for maximum flexibility, as Hinglish
        # queries can mix and match words freely.
        self.intent_keywords = {
            "irrigation": [
                "irrigate", "irrigation", "water", "watering", "paani", "sichai", 
                "sinchai", "drip", "sprinkler", "moisture", "nami", "sookha", "drought"
            ],
            "fertilizer": [
                "fertilizer", "fertiliser", "khaad", "urvarak", "nutrients", "dap", 
                "urea", "npk", "compost", "poshan"
            ],
            "pest_control": [
                "pest", "pests", "insect", "insects", "disease", "keet", "keeda", 
                "rog", "bimari", "spray", "pesticide"
            ],
            "weather": [
                "weather", "forecast", "rain", "temperature", "humidity", "mausam", 
                "baarish", "tapman", "garmi", "sardi"
            ],
            "crop_recommendation": [
                "recommend", "suggest", "grow", "plant", "fasal", "ugaana", "kaun si fasal",
                "which crop", "best crop"
            ],
            "harvest": [
                "harvest", "katai", "fasal katna", "harvesting", "reap", "reaping"
            ]
        }
        
        # --- A list of common crop names in multiple languages ---
        self.crop_entities = [
            "wheat", "gehu", "rice", "dhaan", "chawal", "maize", "makka", "sugarcane", 
            "ganna", "cotton", "kapas", "soybean", "groundnut", "moongfali", "bajra", 
            "barley", "jau", "tea", "chai", "tobacco", "tambaku"
        ]

    async def extract_intent_and_entities(self, query: str, lang: str = "en") -> Dict[str, Any]:
        """
        Extracts intent and entities from the user's query using a reliable
        keyword-spotting method.
        """
        query_lower = query.lower()
        
        # --- Intent Extraction ---
        # Find the intent with the most matching keywords in the query.
        detected_intent = "default"
        best_score = 0
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                detected_intent = intent
        
        # --- Entity Extraction ---
        detected_entities = {}
        detected_crops = []
        for crop in self.crop_entities:
            if crop in query_lower:
                detected_crops.append(crop)
        
        if detected_crops:
            # Use set to remove duplicates if a crop is mentioned twice (e.g., "wheat gehu")
            detected_entities["crop"] = list(set(detected_crops))

        result = {
            "intent": detected_intent,
            "entities": detected_entities,
            "confidence": 0.9 if best_score > 0 else 0.3 # Simple confidence
        }
        
        logger.info(f"NLU Result for query '{query}': {result}")
        return result