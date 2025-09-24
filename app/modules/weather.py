import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from config import config

logger = logging.getLogger(__name__)

class WeatherService:
    """Weather service using OpenWeatherMap API"""
    
    def __init__(self):
        self.api_key = config.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_duration = config.WEATHER_CACHE_DURATION
        self.is_available = bool(self.api_key)
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured")
    
    async def get_weather_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get weather forecast for a location
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary
        """
        try:
            # Check cache first
            cache_key = f"{lat:.4f}_{lon:.4f}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                logger.info(f"Using cached weather data for {lat}, {lon}")
                return cached_data
            
            # Fetch fresh data
            weather_data = await self._fetch_weather_data(lat, lon)
            
            # Cache the data
            self._cache_data(cache_key, weather_data)
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            # Return fallback data
            return self._get_fallback_weather_data(lat, lon)
    
    async def _fetch_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch weather data from OpenWeatherMap API"""
        if not self.api_key:
            raise Exception("OpenWeatherMap API key not configured")
        
        async with httpx.AsyncClient() as client:
            # Get current weather
            current_url = f"{self.base_url}/weather"
            current_params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "lang": "en"
            }
            
            current_response = await client.get(current_url, params=current_params)
            current_response.raise_for_status()
            current_data = current_response.json()
            
            # Get 7-day forecast
            forecast_url = f"{self.base_url}/forecast"
            forecast_params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "lang": "en"
            }
            
            forecast_response = await client.get(forecast_url, params=forecast_params)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            # Process and structure the data
            processed_data = self._process_weather_data(current_data, forecast_data)
            
            return processed_data
    
    def _process_weather_data(self, current_data: Dict, forecast_data: Dict) -> Dict[str, Any]:
        """Process raw weather data into structured format"""
        try:
            # Extract current weather
            current = {
                "temperature": current_data["main"]["temp"],
                "feels_like": current_data["main"]["feels_like"],
                "humidity": current_data["main"]["humidity"],
                "pressure": current_data["main"]["pressure"],
                "description": current_data["weather"][0]["description"],
                "icon": current_data["weather"][0]["icon"],
                "wind_speed": current_data["wind"]["speed"],
                "wind_direction": current_data["wind"].get("deg", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Process forecast data
            daily_forecasts = []
            hourly_forecasts = []
            
            for item in forecast_data["list"]:
                forecast_time = datetime.fromtimestamp(item["dt"])
                
                hourly_forecast = {
                    "timestamp": forecast_time.isoformat(),
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "description": item["weather"][0]["description"],
                    "rain_probability": item.get("pop", 0) * 100,  # Convert to percentage
                    "rain_amount": item.get("rain", {}).get("3h", 0),
                    "wind_speed": item["wind"]["speed"]
                }
                
                hourly_forecasts.append(hourly_forecast)
                
                # Group into daily forecasts
                date_key = forecast_time.date()
                if not daily_forecasts or daily_forecasts[-1]["date"] != date_key.isoformat():
                    daily_forecasts.append({
                        "date": date_key.isoformat(),
                        "max_temp": item["main"]["temp"],
                        "min_temp": item["main"]["temp"],
                        "avg_humidity": item["main"]["humidity"],
                        "total_rain": item.get("rain", {}).get("3h", 0),
                        "rain_probability": item.get("pop", 0) * 100,
                        "description": item["weather"][0]["description"]
                    })
                else:
                    # Update daily stats
                    daily_forecasts[-1]["max_temp"] = max(daily_forecasts[-1]["max_temp"], item["main"]["temp"])
                    daily_forecasts[-1]["min_temp"] = min(daily_forecasts[-1]["min_temp"], item["main"]["temp"])
                    daily_forecasts[-1]["avg_humidity"] = (daily_forecasts[-1]["avg_humidity"] + item["main"]["humidity"]) / 2
                    daily_forecasts[-1]["total_rain"] += item.get("rain", {}).get("3h", 0)
                    daily_forecasts[-1]["rain_probability"] = max(daily_forecasts[-1]["rain_probability"], item.get("pop", 0) * 100)
            
            # Calculate irrigation-relevant metrics
            irrigation_metrics = self._calculate_irrigation_metrics(daily_forecasts, current)
            
            return {
                "current": current,
                "daily_forecast": daily_forecasts[:7],  # 7 days
                "hourly_forecast": hourly_forecasts[:24],  # 24 hours
                "irrigation_metrics": irrigation_metrics,
                "location": {
                    "lat": current_data["coord"]["lat"],
                    "lon": current_data["coord"]["lon"],
                    "name": current_data["name"],
                    "country": current_data["sys"]["country"]
                },
                "fetched_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
            raise Exception(f"Failed to process weather data: {str(e)}")
    
    def _calculate_irrigation_metrics(self, daily_forecasts: list, current: Dict) -> Dict[str, Any]:
        """Calculate metrics relevant for irrigation decisions"""
        try:
            # Rain probability in next 24 hours
            next_24h_rain_prob = max([f.get("rain_probability", 0) for f in daily_forecasts[:1]], default=0)
            
            # Rain probability in next 48 hours
            next_48h_rain_prob = max([f.get("rain_probability", 0) for f in daily_forecasts[:2]], default=0)
            
            # Total expected rain in next 7 days
            total_rain_7d = sum([f.get("total_rain", 0) for f in daily_forecasts[:7]])
            
            # Evapotranspiration estimate (simplified)
            avg_temp = sum([f.get("max_temp", 0) + f.get("min_temp", 0) for f in daily_forecasts[:7]]) / (len(daily_forecasts[:7]) * 2)
            avg_humidity = sum([f.get("avg_humidity", 0) for f in daily_forecasts[:7]]) / len(daily_forecasts[:7])
            
            # Simple ET calculation (Hargreaves equation approximation)
            et_daily = max(0, 0.0023 * (avg_temp + 17.8) * (avg_temp - avg_temp * 0.5) * (1 - avg_humidity / 100))
            
            return {
                "rain_probability_24h": round(next_24h_rain_prob, 1),
                "rain_probability_48h": round(next_48h_rain_prob, 1),
                "total_rain_7d": round(total_rain_7d, 1),
                "evapotranspiration_daily": round(et_daily, 2),
                "irrigation_recommendation": self._get_irrigation_recommendation(
                    next_24h_rain_prob, total_rain_7d, et_daily
                ),
                "risk_factors": self._identify_risk_factors(daily_forecasts, current)
            }
            
        except Exception as e:
            logger.error(f"Error calculating irrigation metrics: {e}")
            return {}
    
    def _get_irrigation_recommendation(self, rain_prob_24h: float, total_rain_7d: float, et_daily: float) -> str:
        """Get irrigation recommendation based on weather"""
        if rain_prob_24h > 60:
            return "skip_irrigation"
        elif rain_prob_24h > 30:
            return "reduce_irrigation"
        elif total_rain_7d < 10 and et_daily > 5:
            return "increase_irrigation"
        else:
            return "normal_irrigation"
    
    def _identify_risk_factors(self, daily_forecasts: list, current: Dict) -> list:
        """Identify weather-related risk factors"""
        risks = []
        
        # High temperature risk
        if current["temperature"] > 35:
            risks.append("high_temperature")
        
        # Low humidity risk
        if current["humidity"] < 30:
            risks.append("low_humidity")
        
        # High wind risk
        if current["wind_speed"] > 20:
            risks.append("high_wind")
        
        # Drought risk
        if all(f.get("total_rain", 0) < 5 for f in daily_forecasts[:3]):
            risks.append("drought_risk")
        
        return risks
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Get cached weather data if still valid"""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if datetime.now() - cached_item["timestamp"] < timedelta(seconds=self.cache_duration):
                return cached_item["data"]
            else:
                # Remove expired cache
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: Dict):
        """Cache weather data"""
        self.cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        # Clean up old cache entries
        current_time = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item["timestamp"] > timedelta(seconds=self.cache_duration)
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _get_fallback_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get fallback weather data when API is unavailable"""
        logger.warning(f"Using fallback weather data for {lat}, {lon}")
        
        return {
            "current": {
                "temperature": 25.0,
                "humidity": 60.0,
                "description": "Unknown",
                "timestamp": datetime.now().isoformat()
            },
            "daily_forecast": [],
            "hourly_forecast": [],
            "irrigation_metrics": {
                "rain_probability_24h": 0.0,
                "irrigation_recommendation": "unknown",
                "risk_factors": ["data_unavailable"]
            },
            "location": {"lat": lat, "lon": lon, "name": "Unknown"},
            "fetched_at": datetime.now().isoformat(),
            "fallback": True
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the weather cache"""
        return {
            "cache_size": len(self.cache),
            "cache_duration": self.cache_duration,
            "cached_locations": list(self.cache.keys())
        }
    
    def clear_cache(self):
        """Clear the weather cache"""
        self.cache.clear()
        logger.info("Weather cache cleared")
