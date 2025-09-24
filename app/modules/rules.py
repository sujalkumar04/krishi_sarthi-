# in app/modules/rules.py

import logging
from typing import Dict, Any
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class IrrigationRuleEngine:
    """
    A streamlined rule engine for irrigation decisions that works directly with
    data from WeatherService and the upgraded SoilDataService.
    """
    
    def __init__(self):
        self.crop_thresholds = config.CROP_IRRIGATION_THRESHOLDS
        
    def apply_rules(self, weather_data: Dict[str, Any], soil_data: Dict[str, Any], 
                   intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply irrigation rules to determine a recommendation.
        """
        try:
            # --- 1. Extract and Estimate Critical Data ---
            
            # Get real weather metrics
            rain_prob_24h = weather_data.get("irrigation_metrics", {}).get("rain_probability_24h", 0.0)
            temperature = weather_data.get("current", {}).get("temperature", 25.0)

            # Get real soil properties
            soil_type = soil_data.get("soil_properties", {}).get("soil_type", "loamy")

            # Get crop info
            crop_type = intent_result.get("entities", {}).get("crop", ["default"])[0]
            crop_thresholds = self.crop_thresholds.get(crop_type.lower(), self.crop_thresholds["default"])
            
            # Estimate soil moisture since we don't have a live sensor value
            estimated_moisture = self._estimate_soil_moisture(weather_data, soil_type)

            # --- 2. Apply Core Rules ---
            
            # Rule 1: High chance of rain? If so, stop and recommend skipping.
            if rain_prob_24h > 60:
                decision = "skip_irrigation"
                reason = f"High rain probability in next 24h ({rain_prob_24h:.1f}%)"
                priority = "low"
            # Rule 2: Is the estimated moisture below the crop's critical threshold?
            elif estimated_moisture <= crop_thresholds["critical"]:
                decision = "irrigate_immediately"
                reason = f"Estimated soil moisture ({estimated_moisture:.1f}%) is below the critical threshold for {crop_type} ({crop_thresholds['critical']}%)"
                priority = "high"
            # Rule 3: Is the estimated moisture below the crop's optimal threshold?
            elif estimated_moisture <= crop_thresholds["optimal"]:
                decision = "irrigate_soon"
                reason = f"Estimated soil moisture ({estimated_moisture:.1f}%) is below the optimal level for {crop_type} ({crop_thresholds['optimal']}%)"
                priority = "medium"
            # Rule 4: If none of the above, moisture is adequate.
            else:
                decision = "monitor"
                reason = f"Estimated soil moisture ({estimated_moisture:.1f}%) is adequate for {crop_type}."
                priority = "low"

            logger.info(f"Rule Engine Decision: {decision}. Reason: {reason}")
            
            return {
                "final_decision": decision,
                "primary_reason": reason,
                "priority": priority,
                "estimated_soil_moisture": estimated_moisture,
                "crop_thresholds_used": crop_thresholds,
                "confidence": 0.8, # Confidence is high because the logic is clear
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error applying irrigation rules: {e}", exc_info=True)
            return self._get_fallback_rule_output()

    def _estimate_soil_moisture(self, weather_data: Dict[str, Any], soil_type: str) -> float:
        """
        A simple heuristic to estimate current soil moisture based on recent weather.
        This is a placeholder for a real soil moisture model or sensor data.
        """
        temp = weather_data.get("current", {}).get("temperature", 25.0)
        humidity = weather_data.get("current", {}).get("humidity", 60.0)
        total_rain_7d = weather_data.get("irrigation_metrics", {}).get("total_rain_7d", 0.0)

        # Start with a base moisture level from the soil's properties
        base_moisture = {"sandy": 12.0, "loamy": 25.0, "clay": 35.0, "alluvial": 28.0}.get(soil_type.lower(), 25.0)

        # Adjust based on rain
        if total_rain_7d > 20:
            moisture_adjustment = 5.0 # Lots of rain recently
        elif total_rain_7d > 5:
            moisture_adjustment = 2.0 # Some rain
        else:
            moisture_adjustment = -5.0 # No rain

        # Adjust based on heat (evaporation)
        if temp > 35:
            moisture_adjustment -= 5.0
        elif temp > 30:
            moisture_adjustment -= 2.0
        
        # Adjust based on humidity
        if humidity < 40:
            moisture_adjustment -= 2.0

        estimated = base_moisture + moisture_adjustment
        # Ensure moisture is within a realistic range (e.g., 5% to 40%)
        return max(5.0, min(40.0, estimated))

    def _get_fallback_rule_output(self) -> Dict[str, Any]:
        """Get fallback rule output when rules fail."""
        return {
            "final_decision": "monitor",
            "primary_reason": "Could not evaluate rules due to an internal error.",
            "priority": "low",
            "confidence": 0.3,
            "timestamp": datetime.now().isoformat()
        }