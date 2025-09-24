# in app/modules/soil.py

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from geopy.distance import geodesic
from config import config

logger = logging.getLogger(__name__)

class SoilDataService:
    """
    Manages and provides soil data from a comprehensive CSV file
    based on geographical location.
    """
    
    def __init__(self):
        self.soil_data_path = config.SOIL_DATA_PATH
        self.soil_df: Optional[pd.DataFrame] = None
        self.is_available = False
        self.last_updated = None
        
        self._load_soil_data()
    
    def _load_soil_data(self):
        """Load soil data from the CSV file."""
        try:
            if not Path(self.soil_data_path).exists():
                logger.error(f"FATAL: Soil data file not found at {self.soil_data_path}. The service will not be available.")
                self.is_available = False
                return

            self.soil_df = pd.read_csv(self.soil_data_path, low_memory=False)
            
            required_cols = ['latitude', 'longitude', 'N', 'P', 'K']
            if not all(col in self.soil_df.columns for col in required_cols):
                logger.error(f"Soil CSV must contain 'latitude', 'longitude', 'N', 'P', and 'K' columns. Please check your file.")
                self.is_available = False
                return

            self.is_available = True
            self.last_updated = datetime.now()
            logger.info(f"Soil data loaded successfully from {self.soil_data_path} with {len(self.soil_df)} rows.")

        except Exception as e:
            logger.error(f"An unexpected error occurred while loading soil data: {e}", exc_info=True)
            self.is_available = False
    
    # --- NEW HELPER FUNCTION TO ESTIMATE SOIL TYPE ---
    def _estimate_soil_type(self, n: float, p: float, k: float) -> str:
        """Estimates soil type based on N, P, K values. This is a heuristic."""
        # These thresholds are examples; you can tune them for better accuracy
        if n > 150 and p > 20 and k > 200:
            return "Alluvial"  # High in all nutrients
        elif p > 40 and k > 40:
            return "Clay" # Often higher in P and K
        elif n < 50 and p < 15 and k < 100:
            return "Sandy" # Generally low in nutrients
        else:
            return "Loamy" # A safe, general-purpose default

    async def get_soil_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get soil data for a specific location by finding the nearest point in the dataset.
        """
        if not self.is_available:
            return self._get_fallback_soil_data(lat, lon, "Service not available")
        
        try:
            distances = self.soil_df.apply(
                lambda row: geodesic((lat, lon), (row['latitude'], row['longitude'])).km,
                axis=1
            )
            
            closest_index = distances.idxmin()
            closest_distance_km = distances.min()
            soil_row = self.soil_df.loc[closest_index]
            
            logger.info(f"Found closest soil data point at index {closest_index} ({closest_distance_km:.2f} km away).")
            return self._process_soil_data(soil_row, lat, lon, closest_distance_km)

        except Exception as e:
            logger.error(f"Error finding nearest soil data for ({lat}, {lon}): {e}", exc_info=True)
            return self._get_fallback_soil_data(lat, lon, "Error during data processing")

    def _process_soil_data(self, soil_row: pd.Series, lat: float, lon: float, distance_km: float) -> Dict[str, Any]:
        """Process the found soil data row into a structured dictionary."""
        
        ph_level = soil_row.get('ph', 7.0)
        nitrogen = soil_row.get('N', 50)
        phosphorus = soil_row.get('P', 50)
        potassium = soil_row.get('K', 50)

        # --- MODIFIED LINE: Call our new helper function ---
        estimated_soil_type = self._estimate_soil_type(nitrogen, phosphorus, potassium)

        return {
            "location": {
                "user_lat": lat,
                "user_lon": lon,
                "data_lat": soil_row.get('latitude'),
                "data_lon": soil_row.get('longitude'),
                "distance_km": round(distance_km, 2),
                "data_source": "interpolated_from_dataset"
            },
            "soil_properties": {
                "soil_type": estimated_soil_type, # <-- USE THE ESTIMATED VALUE
                "ph": ph_level,
                "nitrogen_kg_ha": nitrogen,
                "phosphorus_kg_ha": phosphorus,
                "potassium_kg_ha": potassium,
            },
            "fetched_at": datetime.now().isoformat()
        }

    def _get_fallback_soil_data(self, lat: float, lon: float, reason: str) -> Dict[str, Any]:
        """Return a fallback response when real data cannot be provided."""
        logger.warning(f"Using fallback soil data for ({lat}, {lon}). Reason: {reason}")
        
        return {
            "location": {"lat": lat, "lon": lon, "data_source": "fallback"},
            "soil_properties": {
                "soil_type": "unknown",
                "ph": 7.0,
                "error": "Could not retrieve specific soil data.",
                "reason": reason
            },
            "fetched_at": datetime.now().isoformat(),
            "fallback": True
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the soil data service."""
        return {
            "is_available": self.is_available,
            "data_points": len(self.soil_df) if self.soil_df is not None else 0,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "data_file": self.soil_data_path,
        }