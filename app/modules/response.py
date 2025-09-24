# # # in app/modules/response.py

# # import logging
# # from typing import Dict, Any
# # from pathlib import Path
# # from uuid import uuid4
# # from gtts import gTTS
# # import datetime
# # from config import config

# # logger = logging.getLogger(__name__)

# # class ResponseGenerator:
# #     """
# #     Generates the final user-facing response, including Text-to-Speech (TTS).
# #     """
# #     def __init__(self):
# #         self.audio_output_dir = Path(config.AUDIO_OUTPUT_DIR)
# #         self.tts_engine = None
# #         self._initialize_tts()

# #     def _initialize_tts(self):
# #         """Initializes the gTTS engine."""
# #         try:
# #             # We are using gTTS as the reliable default
# #             self.tts_engine = "gtts"
# #             logger.info("gTTS initialized successfully")
# #         except Exception as e:
# #             logger.error(f"Failed to initialize any TTS engine: {e}")

# #     # --- THE FIX IS IN THIS FUNCTION SIGNATURE ---
# #     # We have removed the 'confidence' argument as it is no longer needed.
# #     async def generate_response(
# #         self, 
# #         parsed_llm_response: Dict[str, Any], 
# #         lang: str
# #     ) -> Dict[str, Any]:
# #         """
# #         Formats the final response and generates an audio file for it.
# #         """
# #         # Safely get the answer, explanation, and sources from the LLM's parsed response
# #         answer = parsed_llm_response.get("answer", "I could not find a specific answer.")
# #         explanation = parsed_llm_response.get("explanation", "")
# #         sources = parsed_llm_response.get("sources", [])

# #         # Generate audio from the primary answer text
# #         audio_url = await self._generate_audio(answer, lang)

# #         return {
# #             "answer": answer,
# #             "explanation": explanation,
# #             "sources": sources,
# #             "audio_url": audio_url
# #         }

# #     async def _generate_audio(self, text: str, lang: str) -> str:
# #         """Generates an audio file from text and returns its web URL."""
# #         if not self.tts_engine or not text:
# #             return None

# #         try:
# #             # Generate a unique filename to prevent browser caching issues
# #             unique_id = uuid4().hex[:8]
# #             filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.mp3"
# #             filepath = self.audio_output_dir / filename

# #             # Use gTTS to create the audio file
# #             tts = gTTS(text=text, lang=lang, slow=False)
# #             tts.save(str(filepath))

# #             # Return the URL that the frontend can use to access the file
# #             audio_url = f"/static/audio/{filename}"
# #             logger.info(f"Generated audio file at: {filepath}")
# #             return audio_url

# #         except Exception as e:
# #             logger.error(f"Error generating TTS audio: {e}")
# #             return None





# import logging
# import datetime  # CORRECT IMPORT
# from typing import Dict, Any
# from pathlib import Path
# from uuid import uuid4
# from gtts import gTTS
# from config import config

# logger = logging.getLogger(__name__)

# class ResponseGenerator:
#     """
#     Generates the final user-facing response, including Text-to-Speech (TTS).
#     """
#     def __init__(self):
#         self.audio_output_dir = Path(config.AUDIO_OUTPUT_DIR)
#         self.tts_engine = None
#         self._initialize_tts()

#     def _initialize_tts(self):
#         """Initializes the gTTS engine."""
#         try:
#             self.tts_engine = "gtts"
#             logger.info("gTTS initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize any TTS engine: {e}")

#     async def generate_response(
#         self, 
#         parsed_llm_response: Dict[str, Any], 
#         lang: str
#     ) -> Dict[str, Any]:
#         """
#         Formats the final response and generates an audio file for it.
#         """
#         answer = parsed_llm_response.get("answer", "I could not find a specific answer.")
#         explanation = parsed_llm_response.get("explanation", "")
#         sources = parsed_llm_response.get("sources", [])

#         audio_url = await self._generate_audio(answer, lang)

#         return {
#             "answer": answer,
#             "explanation": explanation,
#             "sources": sources,
#             "audio_url": audio_url
#         }

#     async def _generate_audio(self, text: str, lang: str) -> str:
#         """Generates an audio file from text and returns its web URL."""
#         if not self.tts_engine or not text:
#             return None

#         try:
#             unique_id = uuid4().hex[:8]
#             # CORRECT DATETIME CALL: datetime.datetime.now()
#             timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f"response_{timestamp}_{unique_id}.mp3"
#             filepath = self.audio_output_dir / filename

#             tts = gTTS(text=text, lang=lang, slow=False)
#             tts.save(str(filepath))

#             audio_url = f"/static/audio/{filename}"
#             logger.info(f"Generated audio file at: {filepath}")
#             return audio_url

#         except Exception as e:
#             # The error message will now be more helpful
#             logger.error(f"Error generating TTS audio: {e}", exc_info=True)
#             return None


# in app/modules/response.py

import logging
import datetime
from typing import Dict, Any
from pathlib import Path
from uuid import uuid4
from gtts import gTTS
from config import config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates the final user-facing response, with optional Text-to-Speech (TTS).
    """
    def __init__(self):
        self.audio_output_dir = Path(config.AUDIO_OUTPUT_DIR)
        self.tts_engine = "gtts"
        logger.info("gTTS initialized successfully")

    async def generate_response(
        self, 
        parsed_llm_response: Dict[str, Any], 
        lang: str,
        generate_audio: bool = False  # <-- NEW: Flag to control audio generation
    ) -> Dict[str, Any]:
        """
        Formats the final response and optionally generates an audio file.
        """
        answer = parsed_llm_response.get("answer", "I could not find a specific answer.")
        explanation = parsed_llm_response.get("explanation", "")
        sources = parsed_llm_response.get("sources", [])
        
        audio_url = None
        # --- THE FIX: Only generate audio if the flag is True ---
        if generate_audio:
            audio_url = await self._generate_audio(answer, lang)

        return {
            "answer": answer,
            "explanation": explanation,
            "sources": sources,
            "audio_url": audio_url
        }

    async def _generate_audio(self, text: str, lang: str) -> str:
        """Generates an audio file from text and returns its web URL."""
        if not text:
            return None
        try:
            unique_id = uuid4().hex[:8]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"response_{timestamp}_{unique_id}.mp3"
            filepath = self.audio_output_dir / filename

            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(filepath))

            audio_url = f"/static/audio/{filename}"
            logger.info(f"Generated audio file at: {filepath}")
            return audio_url
        except Exception as e:
            logger.error(f"Error generating TTS audio: {e}", exc_info=True)
            return None