# import whisper
# import tempfile
# import os
# from pathlib import Path
# from typing import Optional
# import logging
# from config import config

# logger = logging.getLogger(__name__)

# class SpeechToText:
#     """Speech-to-Text service using OpenAI Whisper"""
    
#     def __init__(self):
#         self.model = None
#         self.is_initialized = False
        
#     async def initialize(self):
#         """Initialize the Whisper model"""
#         try:
#             # Load the appropriate model based on language
#             model_size = "base"  # Can be "tiny", "base", "small", "medium", "large"
#             self.model = whisper.load_model(model_size)
#             self.is_initialized = True
#             logger.info(f"Whisper model {model_size} loaded successfully")
#         except Exception as e:
#             logger.error(f"Failed to load Whisper model: {e}")
#             self.is_initialized = False
    
#     async def transcribe_audio(self, audio_file, language: str = "hi") -> str:
#         """
#         Transcribe audio file to text
        
#         Args:
#             audio_file: Uploaded audio file
#             language: Expected language (hi/en)
            
#         Returns:
#             Transcribed text
#         """
#         if not self.is_initialized:
#             await self.initialize()
            
#         try:
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
#                 content = await audio_file.read()
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name
            
#             # Transcribe using Whisper
#             result = self.model.transcribe(
#                 temp_file_path,
#                 language=language if language != "hinglish" else "hi",
#                 task="transcribe"
#             )
            
#             # Clean up temporary file
#             os.unlink(temp_file_path)
            
#             transcribed_text = result["text"].strip()
#             logger.info(f"Transcribed audio: {transcribed_text[:100]}...")
            
#             return transcribed_text
            
#         except Exception as e:
#             logger.error(f"Error transcribing audio: {e}")
#             raise Exception(f"Failed to transcribe audio: {str(e)}")
    
#     async def transcribe_file(self, file_path: str, language: str = "hi") -> str:
#         """
#         Transcribe audio file from path
        
#         Args:
#             file_path: Path to audio file
#             language: Expected language
            
#         Returns:
#             Transcribed text
#         """
#         if not self.is_initialized:
#             await self.initialize()
            
#         try:
#             result = self.model.transcribe(
#                 file_path,
#                 language=language if language != "hinglish" else "hi",
#                 task="transcribe"
#             )
            
#             return result["text"].strip()
            
#         except Exception as e:
#             logger.error(f"Error transcribing file {file_path}: {e}")
#             raise Exception(f"Failed to transcribe file: {str(e)}")
    
#     def get_supported_languages(self) -> list:
#         """Get list of supported languages"""
#         return ["hi", "en", "hinglish"]
    
#     def get_model_info(self) -> dict:
#         """Get information about the loaded model"""
#         if self.model:
#             return {
#                 "model_name": self.model.name,
#                 "model_size": self.model.dims,
#                 "is_initialized": self.is_initialized
#             }
#         return {"is_initialized": False}

# in app/modules/stt.py

import logging
import whisper
import tempfile
import os
from fastapi import UploadFile

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Speech-to-Text service using OpenAI's Whisper model.
    """
    def __init__(self, model_name: str = "base"):
        self.model = None
        self.model_name = model_name
        try:
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)

    async def transcribe_audio(self, audio_file: UploadFile, lang: str) -> str:
        """
        Transcribes an uploaded audio file to text.
        This function now correctly saves the uploaded file to a temporary location
        before passing it to Whisper.
        """
        if not self.model:
            logger.error("Whisper model is not available. Cannot transcribe.")
            return "Speech-to-text service is unavailable."

        tmp_path = None
        try:
            # Create a temporary file to store the uploaded audio content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                # --- THE CORE FIX ---
                # 1. Read the binary content from the uploaded file stream.
                contents = await audio_file.read()
                # 2. Write that content to the temporary file on disk.
                tmp.write(contents)
                tmp_path = tmp.name
            
            logger.info(f"Uploaded audio saved to temporary file: {tmp_path}")

            # Transcribe the audio file using the specified language
            # Forcing language to 'en' if Hinglish is selected, as Whisper is best at English
            transcription_lang = 'en' if lang == 'hinglish' else lang
            result = self.model.transcribe(tmp_path, language=transcription_lang, fp16=False)
            
            transcribed_text = result.get("text", "").strip()
            logger.info(f"Transcription successful: '{transcribed_text}'")
            return transcribed_text

        except Exception as e:
            logger.error(f"Error during audio transcription: {e}", exc_info=True)
            return "Error during audio transcription."
        finally:
            # --- ROBUST CLEANUP ---
            # Ensure the temporary file is always deleted after transcription
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
