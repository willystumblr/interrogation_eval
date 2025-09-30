import json
import re
from typing import Dict, Any
import requests
from pydantic import BaseModel, Field
import logging

class GoogleGeocodeValidate(BaseModel):
    """LiteLLM-compatible tool that marks an address **valid** only when the Google
    Maps Geocoding API returns a *precise* (ROOFTOP) result whose formatted
    address matches the user-supplied string (ignoring punctuation/case).
    Anything else—including partial matches, approximate results, or API
    failures—is **invalid**. """

    api_key: str = Field(..., description="Google Maps Geocoding API key")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _geocode(self, address: str) -> Dict[str, Any]:
        """Call Google Geocoding API and return the raw JSON response."""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": self.api_key}
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        return resp.json()
    # ---------------------------------------------------------------------
    # Tool entry point
    # ---------------------------------------------------------------------

    def invoke(self, address: str) -> str:  # noqa: D401  (command verbs fine)
        """Return 'valid' when a rooftop-level exact match exists; else 'invalid'."""
        try:
            data = self._geocode(address)

            if data.get("status") != "OK":
                # Includes ZERO_RESULTS, OVER_QUERY_LIMIT, etc.
                raise ValueError(f"Google API error: {data.get('status')}")

            for res in data.get("results", []): # assuming only one result is needed
                """
                > Generally, only one entry in the "results" array is returned for address lookups, though the geocoder may return several results when address queries are ambiguous.
                > _https://developers.google.com/maps/documentation/geocoding/requests-geocoding_
                """
                if res.get("partial_match"):
                    # Google explicitly says this is an inexact match.
                    logging.info(f"Partial match: {res.get('formatted_address')} against {address}")
                    return json.dumps([{"location_type": None, "formatted_address":res.get('formatted_address'), "matched_address": None, "lat": None, "lng": None}])
                loc_type = res.get("geometry", {}).get("location_type", "")
                if not loc_type:
                    # No location type means no precise match.
                    logging.info(f"No location type for: {res.get('formatted_address')} against {address}")
                    return json.dumps([{"location_type": None, "formatted_address":res.get('formatted_address'), "matched_address": None, "lat": None, "lng": None}])
                
                formatted = res.get("formatted_address", "")
                logging.info(f"{loc_type}: <{formatted}> against <{address}>")
                return json.dumps([{"location_type": loc_type, "formatted_address": formatted, "lat": res.get("geometry", {}).get("location", {}).get("lat"), "lng": res.get("geometry", {}).get("location", {}).get("lng")}])
            

        except Exception:
            # Network errors, invalid key, etc. – fail closed.
            return json.dumps([{"location_type": None, "matched_address": None, "lat": None, "lng": None}])

    # ---------------------------------------------------------------------
    # Schema for LiteLLM
    # ---------------------------------------------------------------------

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """Return JSON schema for LiteLLM registration."""
        return {
            "type": "function",
            "function": {
                "name": "google_geocode_validate",
                "description": (
                    "Verify an address by querying the Google Maps Geocoding API. "
                    "The API returns a `location_type` to indicate how precise or reliable the location is:\n"
                    "- `ROOFTOP`: Exact match, highly reliable.\n"
                    "- `RANGE_INTERPOLATED`: Approximate match, less reliable.\n"
                    "- `GEOMETRIC_CENTER`: Geometric center of a result set, not a precise location.\n"
                    "- `APPROXIMATE`: Approximate location, low reliability."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "string",
                            "description": "Full postal address to validate."
                        }
                    },
                    "required": ["address"],
                },
            },
        }


