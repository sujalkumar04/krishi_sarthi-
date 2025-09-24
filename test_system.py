#!/usr/bin/env python3
"""
Test script for Agriculture AI Assistant
Tests all major components to ensure they're working correctly
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

from config import config
from app.modules.stt import SpeechToText
from app.modules.nlu import IntentExtractor
from app.modules.weather import WeatherService
from app.modules.soil import SoilDataService
from app.modules.rules import IrrigationRuleEngine
from app.modules.rag import RAGService
from app.modules.llm_local import LocalLLMService
from app.modules.llm_cloud import CloudLLMService
from app.modules.response import ResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """Test all system components"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all system tests"""
        logger.info("🚀 Starting Agriculture AI Assistant System Tests")
        
        try:
            # Test 1: Configuration
            await self.test_configuration()
            
            # Test 2: STT Service
            await self.test_stt_service()
            
            # Test 3: NLU Service
            await self.test_nlu_service()
            
            # Test 4: Weather Service
            await self.test_weather_service()
            
            # Test 5: Soil Service
            await self.test_soil_service()
            
            # Test 6: Rule Engine
            await self.test_rule_engine()
            
            # Test 7: RAG Service
            await self.test_rag_service()
            
            # Test 8: LLM Services
            await self.test_llm_services()
            
            # Test 9: Response Generator
            await self.test_response_generator()
            
            # Test 10: End-to-End Pipeline
            await self.test_end_to_end()
            
            # Print results
            self.print_results()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise
    
    async def test_configuration(self):
        """Test configuration loading"""
        logger.info("🔧 Testing Configuration...")
        
        try:
            # Check if config loaded
            assert config is not None, "Config not loaded"
            
            # Check required settings
            assert hasattr(config, 'OPENWEATHER_API_KEY'), "Missing OpenWeather API key"
            assert hasattr(config, 'OPENAI_API_KEY'), "Missing OpenAI API key"
            assert hasattr(config, 'LOCAL_LLM_PATH'), "Missing local LLM path"
            
            # Check crop thresholds
            assert len(config.CROP_IRRIGATION_THRESHOLDS) > 0, "No crop thresholds defined"
            
            # Create directories
            config.create_directories()
            
            self.test_results['configuration'] = {'status': 'PASS', 'message': 'Configuration loaded successfully'}
            logger.info("✅ Configuration test passed")
            
        except Exception as e:
            self.test_results['configuration'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ Configuration test failed: {e}")
    
    async def test_stt_service(self):
        """Test Speech-to-Text service"""
        logger.info("🎤 Testing STT Service...")
        
        try:
            stt = SpeechToText()
            await stt.initialize()
            
            # Check if initialized
            assert stt.is_initialized, "STT service not initialized"
            
            # Check supported languages
            languages = stt.get_supported_languages()
            assert len(languages) > 0, "No supported languages"
            
            self.test_results['stt_service'] = {'status': 'PASS', 'message': 'STT service working'}
            logger.info("✅ STT service test passed")
            
        except Exception as e:
            self.test_results['stt_service'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ STT service test failed: {e}")
    
    async def test_nlu_service(self):
        """Test Natural Language Understanding service"""
        logger.info("🧠 Testing NLU Service...")
        
        try:
            nlu = IntentExtractor()
            
            # Test intent extraction
            test_queries = [
                "Should I irrigate my wheat crop today?",
                "क्या मुझे आज गेहूं की फसल में पानी देना चाहिए?",
                "When should I apply fertilizer to rice?",
                "मौसम कैसा है?"
            ]
            
            for query in test_queries:
                result = await nlu.extract_intent_and_entities(query)
                assert 'intent' in result, f"No intent found for: {query}"
                assert 'confidence' in result, f"No confidence found for: {query}"
            
            self.test_results['nlu_service'] = {'status': 'PASS', 'message': 'NLU service working'}
            logger.info("✅ NLU service test passed")
            
        except Exception as e:
            self.test_results['nlu_service'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ NLU service test failed: {e}")
    
    async def test_weather_service(self):
        """Test Weather service"""
        logger.info("🌤️ Testing Weather Service...")
        
        try:
            weather = WeatherService()
            
            # Check if API key configured
            if not weather.is_available:
                logger.warning("⚠️ Weather service not available (no API key)")
                self.test_results['weather_service'] = {'status': 'SKIP', 'message': 'No API key configured'}
                return
            
            # Test weather fetch (use default coordinates)
            weather_data = await weather.get_weather_forecast(
                config.DEFAULT_LOCATION_LAT, 
                config.DEFAULT_LOCATION_LON
            )
            
            assert 'current' in weather_data, "No current weather data"
            assert 'irrigation_metrics' in weather_data, "No irrigation metrics"
            
            self.test_results['weather_service'] = {'status': 'PASS', 'message': 'Weather service working'}
            logger.info("✅ Weather service test passed")
            
        except Exception as e:
            self.test_results['weather_service'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ Weather service test failed: {e}")
    
    async def test_soil_service(self):
        """Test Soil service"""
        logger.info("🌱 Testing Soil Service...")
        
        try:
            soil = SoilDataService()
            
            # Test soil data retrieval
            soil_data = await soil.get_soil_data(
                config.DEFAULT_LOCATION_LAT, 
                config.DEFAULT_LOCATION_LON
            )
            
            assert 'soil_properties' in soil_data, "No soil properties"
            assert 'current_status' in soil_data, "No current status"
            assert 'irrigation_analysis' in soil_data, "No irrigation analysis"
            
            self.test_results['soil_service'] = {'status': 'PASS', 'message': 'Soil service working'}
            logger.info("✅ Soil service test passed")
            
        except Exception as e:
            self.test_results['soil_service'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ Soil service test failed: {e}")
    
    async def test_rule_engine(self):
        """Test Rule Engine"""
        logger.info("⚙️ Testing Rule Engine...")
        
        try:
            rules = IrrigationRuleEngine()
            
            # Test with sample data
            weather_data = {
                "current": {"temperature": 25.0, "humidity": 60.0},
                "irrigation_metrics": {
                    "rain_probability_24h": 20.0,
                    "rain_probability_48h": 30.0,
                    "total_rain_7d": 5.0,
                    "evapotranspiration_daily": 4.0
                }
            }
            
            soil_data = {
                "current_status": {"moisture_percentage": 18.0},
                "soil_properties": {"field_capacity": 25.0, "wilting_point": 8.0}
            }
            
            intent_result = {"intent": "irrigation", "entities": {"crop": ["wheat"]}}
            
            rule_output = rules.apply_rules(weather_data, soil_data, intent_result)
            
            assert 'final_decision' in rule_output, "No decision from rule engine"
            assert 'confidence' in rule_output, "No confidence from rule engine"
            
            self.test_results['rule_engine'] = {'status': 'PASS', 'message': 'Rule engine working'}
            logger.info("✅ Rule engine test passed")
            
        except Exception as e:
            self.test_results['rule_engine'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ Rule engine test failed: {e}")
    
    async def test_rag_service(self):
        """Test RAG service"""
        logger.info("📚 Testing RAG Service...")
        
        try:
            rag = RAGService()
            await rag.initialize()
            
            # Check if initialized
            assert rag.is_initialized, "RAG service not initialized"
            
            # Get index stats
            stats = rag.get_index_stats()
            assert 'total_documents' in stats, "No index stats"
            
            self.test_results['rag_service'] = {'status': 'PASS', 'message': 'RAG service working'}
            logger.info("✅ RAG service test passed")
            
        except Exception as e:
            self.test_results['rag_service'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ RAG service test failed: {e}")
    
    async def test_llm_services(self):
        """Test LLM services"""
        logger.info("🤖 Testing LLM Services...")
        
        try:
            # Test local LLM
            local_llm = LocalLLMService()
            await local_llm.initialize()
            
            if local_llm.is_initialized:
                self.test_results['local_llm'] = {'status': 'PASS', 'message': 'Local LLM working'}
                logger.info("✅ Local LLM test passed")
            else:
                self.test_results['local_llm'] = {'status': 'SKIP', 'message': 'Local LLM not available'}
                logger.warning("⚠️ Local LLM not available")
            
            # Test cloud LLM
            cloud_llm = CloudLLMService()
            await cloud_llm.initialize()
            
            if cloud_llm.is_available:
                self.test_results['cloud_llm'] = {'status': 'PASS', 'message': 'Cloud LLM working'}
                logger.info("✅ Cloud LLM test passed")
            else:
                self.test_results['cloud_llm'] = {'status': 'SKIP', 'message': 'Cloud LLM not available'}
                logger.warning("⚠️ Cloud LLM not available")
            
        except Exception as e:
            self.test_results['llm_services'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ LLM services test failed: {e}")
    
    async def test_response_generator(self):
        """Test Response Generator"""
        logger.info("🎵 Testing Response Generator...")
        
        try:
            response_gen = ResponseGenerator()
            
            # Check if initialized
            assert response_gen.is_initialized, "Response generator not initialized"
            
            # Check supported languages
            languages = response_gen.get_supported_tts_languages()
            assert len(languages) > 0, "No supported TTS languages"
            
            self.test_results['response_generator'] = {'status': 'PASS', 'message': 'Response generator working'}
            logger.info("✅ Response generator test passed")
            
        except Exception as e:
            self.test_results['response_generator'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ Response generator test failed: {e}")
    
    async def test_end_to_end(self):
        """Test end-to-end pipeline"""
        logger.info("🔄 Testing End-to-End Pipeline...")
        
        try:
            # This is a simplified test - in real scenarios you'd test the full pipeline
            # For now, we'll just check if all components can work together
            
            # Create sample data
            query = "Should I irrigate my wheat crop today?"
            lat, lon = config.DEFAULT_LOCATION_LAT, config.DEFAULT_LOCATION_LON
            language = "en"
            
            # Test NLU
            nlu = IntentExtractor()
            intent_result = await nlu.extract_intent_and_entities(query, language)
            
            # Test weather (if available)
            weather_data = {}
            if config.OPENWEATHER_API_KEY:
                weather = WeatherService()
                weather_data = await weather.get_weather_forecast(lat, lon)
            
            # Test soil
            soil = SoilDataService()
            soil_data = await soil.get_soil_data(lat, lon)
            
            # Test rules
            rules = IrrigationRuleEngine()
            rule_output = rules.apply_rules(weather_data, soil_data, intent_result)
            
            # Test RAG
            rag = RAGService()
            await rag.initialize()
            retrieved_docs = await rag.retrieve_relevant_docs(query, intent_result)
            
            # If we get here without errors, the pipeline is working
            self.test_results['end_to_end'] = {'status': 'PASS', 'message': 'End-to-end pipeline working'}
            logger.info("✅ End-to-end pipeline test passed")
            
        except Exception as e:
            self.test_results['end_to_end'] = {'status': 'FAIL', 'message': str(e)}
            logger.error(f"❌ End-to-end pipeline test failed: {e}")
    
    def print_results(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("🧪 AGRICULTURE AI ASSISTANT - SYSTEM TEST RESULTS")
        print("="*60)
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, result in self.test_results.items():
            status = result['status']
            message = result['message']
            
            if status == 'PASS':
                print(f"✅ {test_name:20} - PASS")
                passed += 1
            elif status == 'FAIL':
                print(f"❌ {test_name:20} - FAIL: {message}")
                failed += 1
            elif status == 'SKIP':
                print(f"⚠️  {test_name:20} - SKIP: {message}")
                skipped += 1
        
        print("-"*60)
        print(f"📊 SUMMARY: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED")
        
        if failed == 0:
            print("🎉 All critical tests passed! System is ready to use.")
        else:
            print("⚠️  Some tests failed. Please check the configuration and dependencies.")
        
        print("="*60)
        
        # Save results to file
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"📄 Detailed results saved to: {results_file}")

async def main():
    """Main test function"""
    try:
        tester = SystemTester()
        await tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
