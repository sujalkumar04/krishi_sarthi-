import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _clean_env(value: str) -> str:
    """Strip inline comments and whitespace from an env value."""
    if value is None:
        return ""
    return value.split('#', 1)[0].strip()


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default))
    val = _clean_env(raw).lower()
    return val in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    cleaned = _clean_env(raw)
    try:
        return int(cleaned) if cleaned != "" else int(default)
    except ValueError:
        return int(default)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    cleaned = _clean_env(raw)
    try:
        return float(cleaned) if cleaned != "" else float(default)
    except ValueError:
        return float(default)


class Config:
    """Configuration class for the Agriculture AI Assistant"""
    
    # API Keys
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    LOCAL_LLM_PATH: str = os.getenv("LOCAL_LLM_PATH", "models/llama-8b-ggml-q4_0.bin")
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    USE_LOCAL_LLM: bool = _get_env_bool("USE_LOCAL_LLM", True)
    LLM_CONFIDENCE_THRESHOLD: float = _get_env_float("LLM_CONFIDENCE_THRESHOLD", 0.7)
    
    # Weather API Configuration
    WEATHER_CACHE_DURATION: int = _get_env_int("WEATHER_CACHE_DURATION", 3600)
    DEFAULT_LOCATION_LAT: float = _get_env_float("DEFAULT_LOCATION_LAT", 28.6139)
    DEFAULT_LOCATION_LON: float = _get_env_float("DEFAULT_LOCATION_LON", 77.2090)
    
    # RAG Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "data/faiss_index")
    CHUNK_SIZE: int = _get_env_int("CHUNK_SIZE", 400)
    CHUNK_OVERLAP: int = _get_env_int("CHUNK_OVERLAP", 50)
    TOP_K_RETRIEVAL: int = _get_env_int("TOP_K_RETRIEVAL", 5)
    
    # Audio Configuration
    AUDIO_OUTPUT_DIR: str = os.getenv("AUDIO_OUTPUT_DIR", "static/audio")
    TTS_LANGUAGE: str = _clean_env(os.getenv("TTS_LANGUAGE", "hi")) or "hi"
    STT_LANGUAGE: str = _clean_env(os.getenv("STT_LANGUAGE", "hi")) or "hi"
    
    # Server Configuration
    HOST: str = _clean_env(os.getenv("HOST", "0.0.0.0")) or "0.0.0.0"
    PORT: int = _get_env_int("PORT", 8000)
    DEBUG: bool = _get_env_bool("DEBUG", True)
    
    # Database Configuration
    SOIL_DATA_PATH: str = os.getenv("SOIL_DATA_PATH", "data/soil_data.csv")
    KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base/")
    
    # Crop-specific irrigation thresholds (mm)
    CROP_IRRIGATION_THRESHOLDS = {
        "wheat": {"critical": 15, "optimal": 25, "excess": 35},
        "rice": {"critical": 20, "optimal": 30, "excess": 40},
        "maize": {"critical": 18, "optimal": 28, "excess": 38},
        "cotton": {"critical": 16, "optimal": 26, "excess": 36},
        "sugarcane": {"critical": 22, "optimal": 32, "excess": 42},
        "default": {"critical": 17, "optimal": 27, "excess": 37}
    }
    
    # Language mappings
    LANGUAGE_NAMES = {
        "hi": "Hindi",
        "en": "English",
        "hinglish": "Hinglish"
    }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.OPENWEATHER_API_KEY:
            print("Warning: OPENWEATHER_API_KEY not set")
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set")
        if not cls.USE_LOCAL_LLM and not cls.OPENAI_API_KEY:
            print("Error: Must have either LOCAL_LLM_PATH or OPENAI_API_KEY")
            return False
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            "data",
            "data/faiss_index",
            "data/knowledge_base",
            "static",
            "static/audio",
            "models"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()
