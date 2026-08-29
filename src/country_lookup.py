#!/usr/bin/env python
"""
Robust country lookup with caching and fallbacks for Maidenhead grids.
Handles geocoding failures gracefully using known grid→country mappings.
"""
import json
import os
from pyhamtools.locator import locator_to_latlong
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Pre-defined grid→country mappings (critical for Egypt/India paths)
KNOWN_GRIDS = {
    # Egypt grids (KM49 is Cairo region)
    'KM49': 'EG', 'KM48': 'EG', 'KM59': 'EG', 'KM58': 'EG',
    'KL49': 'EG', 'KL48': 'EG', 'KL59': 'EG', 'KL58': 'EG',
    # India grids (MK90 is Delhi region)
    'MK90': 'IN', 'MK91': 'IN', 'MK80': 'IN', 'MK81': 'IN',
    'ML90': 'IN', 'ML91': 'IN', 'ML80': 'IN', 'ML81': 'IN',
    'NJ90': 'IN', 'NJ91': 'IN', 'NJ80': 'IN', 'NJ81': 'IN',
    # UK grids (Leeds = IO93)
    'IO93': 'GB', 'IO94': 'GB', 'IO83': 'GB', 'IO84': 'GB',
    # Common European grids
    'JO': 'DE', 'JN': 'IT', 'IN': 'PT', 'IM': 'ES', 'KO': 'RU',
    'KP': 'UA', 'KN': 'BG', 'KM': 'GR', 'IL': 'FR', 'IO': 'GB'
}

class CountryLookup:
    def __init__(self, cache_file='country_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.geolocator = Nominatim(user_agent="skywave_lookup", timeout=5)
        print(f"Loaded {len(self.cache)} cached grid→country mappings")
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def grid_to_country(self, grid):
        """Convert Maidenhead grid to country code with fallbacks"""
        if not grid or len(grid) < 4:
            return "UNKNOWN"
        
        # Use first 4 characters (e.g., KM49)
        grid4 = grid[:4].upper()
        
        # 1. Check pre-defined mappings (fastest)
        if grid4 in KNOWN_GRIDS:
            return KNOWN_GRIDS[grid4]
        
        # 2. Check cache
        if grid4 in self.cache:
            return self.cache[grid4]
        
        # 3. Try geocoding (slow, may fail)
        try:
            lat, lon = locator_to_latlong(grid4)
            location = self.geolocator.reverse((lat, lon), language='en', timeout=3)
            country_code = location.raw['address']['country_code'].upper()
            self.cache[grid4] = country_code
            self._save_cache()
            return country_code
        except (GeocoderTimedOut, GeocoderServiceError, KeyError, Exception) as e:
            # Fallback: use first two letters for rough region
            prefix = grid4[:2]
            if prefix in ['KM', 'KL']:  # Egypt region
                self.cache[grid4] = 'EG'
                self._save_cache()
                return 'EG'
            elif prefix in ['MK', 'ML', 'NJ']:  # India region
                self.cache[grid4] = 'IN'
                self._save_cache()
                return 'IN'
            else:
                self.cache[grid4] = 'UNKNOWN'
                self._save_cache()
                return 'UNKNOWN'

# Initialize singleton
lookup = CountryLookup()

def grid_to_country(grid):
    return lookup.grid_to_country(grid)