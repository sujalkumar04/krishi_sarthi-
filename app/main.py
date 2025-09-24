# in app/main.py

# import logging
# import traceback
# from pathlib import Path
# import asyncio
# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse, HTMLResponse
# import uvicorn

# # --- Configure Logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # --- Import All Your Services (CloudLLMService is now removed) ---
# from config import config
# from .modules.stt import SpeechToText
# from .modules.nlu import IntentExtractor
# from .modules.weather import WeatherService
# from .modules.soil import SoilDataService
# from .modules.rules import IrrigationRuleEngine
# from .modules.rag import RAGService
# from .modules.llm_local import LocalLLMService
# # REMOVED: from .modules.llm_cloud import CloudLLMService
# from .modules.response import ResponseGenerator

# # --- Initialize App and Services ---
# try:
#     config.create_directories()
#     app = FastAPI(title="Agriculture AI Assistant", version="1.0.0")

#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
#     )

#     stt_service = SpeechToText()
#     nlu_service = IntentExtractor()
#     weather_service = WeatherService()
#     soil_service = SoilDataService()
#     rule_engine = IrrigationRuleEngine()
#     rag_service = RAGService()
#     local_llm = LocalLLMService()
#     # REMOVED: cloud_llm = CloudLLMService()
#     response_gen = ResponseGenerator()
# except Exception as e:
#     logger.critical(f"CRITICAL: Failed to initialize a core service. Error: {e}", exc_info=True)
#     exit(1)

# # Mount static files to serve the front-end
# static_dir = Path("static")
# if static_dir.is_dir():
#     app.mount("/static", StaticFiles(directory=static_dir), name="static")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize heavy services in the background on startup."""
#     logger.info("🚀 Agriculture AI Assistant is starting up...")
#     asyncio.create_task(rag_service.initialize())
#     asyncio.create_task(local_llm.initialize()) # Always initialize the local LLM
#     logger.info("✅ Server startup initiated. Heavy services are warming up in the background.")

# # --- API Endpoints ---

# @app.get("/", response_class=HTMLResponse)
# async def get_root_page():
#     """Redirects the root URL to the main web interface."""
#     index_path = Path("static/index.html")
#     if index_path.exists():
#         return FileResponse(index_path)
#     return HTMLResponse("<h1>Agriculture AI Assistant API</h1><p>API is running, but the frontend file (static/index.html) was not found.</p>")

# @app.get("/health")
# async def health_check():
#     """Provides a detailed health check of all services."""
#     return {
#         "status": "healthy", 
#         "services": {
#             "rag": rag_service.is_initialized,
#             "local_llm": local_llm.is_initialized, # Simplified health check
#             "weather": weather_service.is_available,
#             "soil": soil_service.is_available
#         }
#     }

# @app.post("/ask")
# async def ask_question(
#     query: str = Form(None),
#     lat: float = Form(...),
#     lon: float = Form(...),
#     lang: str = Form("en"),
#     audio_file: UploadFile = File(None)
# ):
#     """Main endpoint for processing agricultural queries."""
#     logger.info(f"Received /ask request for query: '{query}'")
#     try:
#         if audio_file:
#             query = await stt_service.transcribe_audio(audio_file, lang)
#         if not query:
#             raise HTTPException(status_code=400, detail="Query is empty.")

#         # Run data gathering and rule engine concurrently
#         intent_result, weather_data, soil_data = await asyncio.gather(
#             nlu_service.extract_intent_and_entities(query, lang),
#             weather_service.get_weather_forecast(lat, lon),
#             soil_service.get_soil_data(lat, lon)
#         )
        
#         rule_output = rule_engine.apply_rules(weather_data, soil_data, intent_result)
#         retrieved_docs = await rag_service.retrieve_relevant_docs(query, intent_result)
        
#         # Simplified LLM logic: always use local LLM
#         if not local_llm.is_initialized:
#             logger.warning("Local LLM is not ready yet, attempting to initialize now...")
#             await local_llm.initialize() # Attempt a final initialization
        
#         if local_llm.is_initialized:
#             llm_response = await local_llm.generate_response(query, intent_result, weather_data, soil_data, rule_output, retrieved_docs, lang)
#         else:
#             logger.error("Local LLM failed to initialize. Returning a fallback response.")
#             llm_response = local_llm._get_fallback_response(query, lang)
        
#         parsed_llm_response = llm_response.get("parsed_response", {})
#         final_response = await response_gen.generate_response(parsed_llm_response, lang)

#         return {
#             "answer": final_response.get("answer", "Error: Could not generate a final answer."),
#             "explanation": final_response.get("explanation", ""),
#             "sources": final_response.get("sources", []),
#             "audio_url": final_response.get("audio_url"),
#             "debug_info": {"intent": intent_result, "rule_output": rule_output}
#         }
        
#     except Exception as e:
#         logger.error(f"FATAL ERROR in /ask endpoint: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An unexpected error occurred. Please check the server logs.")

# @app.get("/audio/{filename}")
# async def get_audio(filename: str):
#     audio_path = Path(config.AUDIO_OUTPUT_DIR) / filename
#     if audio_path.exists():
#         return FileResponse(audio_path)
#     raise HTTPException(status_code=404, detail="Audio file not found")

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)


# in app/main.py

import logging
import traceback
from pathlib import Path
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import config
from .modules.stt import SpeechToText
from .modules.nlu import IntentExtractor
from .modules.weather import WeatherService
from .modules.soil import SoilDataService
from .modules.rules import IrrigationRuleEngine
from .modules.rag import RAGService
from .modules.llm_local import LocalLLMService
from .modules.response import ResponseGenerator

try:
    config.create_directories()
    app = FastAPI(title="Agriculture AI Assistant", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    stt_service, nlu_service, weather_service, soil_service, rule_engine, rag_service, local_llm, response_gen = (SpeechToText(), IntentExtractor(), WeatherService(), SoilDataService(), IrrigationRuleEngine(), RAGService(), LocalLLMService(), ResponseGenerator())
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize a core service. Error: {e}", exc_info=True); exit(1)

static_dir = Path("static")
if static_dir.is_dir(): app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Agriculture AI Assistant is starting up...")
    asyncio.create_task(rag_service.initialize())
    asyncio.create_task(local_llm.initialize())
    logger.info("✅ Server startup initiated. Heavy services are warming up in the background.")

@app.get("/", response_class=HTMLResponse)
async def get_root_page():
    index_path = Path("static/index.html")
    return FileResponse(index_path) if index_path.exists() else HTMLResponse("<h1>API is running</h1><p>Frontend file not found.</p>")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {"rag": rag_service.is_initialized, "local_llm": local_llm.is_initialized, "weather": weather_service.is_available, "soil": soil_service.is_available}}

@app.post("/ask")
async def ask_question(
    query: str = Form(None),
    lat: float = Form(...),
    lon: float = Form(...),
    lang: str = Form("en"),
    audio_file: UploadFile = File(None),
    generate_audio: bool = Form(False) # <-- NEW: Receive the toggle state from the UI
):
    logger.info(f"Received /ask request. Query: '{query}', Generate Audio: {generate_audio}")
    try:
        # --- THE FIX: A more robust check for a real audio file ---
        if audio_file and audio_file.filename and audio_file.size > 0:
            logger.info(f"Transcribing audio file: {audio_file.filename}")
            query = await stt_service.transcribe_audio(audio_file, lang)
        if not query:
            raise HTTPException(status_code=400, detail="Query is empty.")

        intent_result, weather_data, soil_data = await asyncio.gather(
            nlu_service.extract_intent_and_entities(query, lang),
            weather_service.get_weather_forecast(lat, lon),
            soil_service.get_soil_data(lat, lon)
        )
        
        rule_output = rule_engine.apply_rules(weather_data, soil_data, intent_result)
        retrieved_docs = await rag_service.retrieve_relevant_docs(query, intent_result)
        
        if local_llm.is_initialized:
            llm_response = await local_llm.generate_response(query, intent_result, weather_data, soil_data, rule_output, retrieved_docs, lang)
        else:
            logger.error("Local LLM not available. Returning fallback.")
            llm_response = local_llm._get_fallback_response(query, lang)
        
        parsed_llm_response = llm_response.get("parsed_response", {})
        
        # --- THE FIX: Pass the audio toggle state to the response generator ---
        final_response = await response_gen.generate_response(
            parsed_llm_response, lang, generate_audio=generate_audio
        )

        return {
            "answer": final_response.get("answer", "Error."),
            "explanation": final_response.get("explanation", ""),
            "sources": final_response.get("sources", []),
            "audio_url": final_response.get("audio_url"),
            "debug_info": {"intent": intent_result, "rule_output": rule_output}
        }
    except Exception as e:
        logger.error(f"FATAL ERROR in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = Path(config.AUDIO_OUTPUT_DIR) / filename
    return FileResponse(audio_path) if audio_path.exists() else HTTPException(status_code=404)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)