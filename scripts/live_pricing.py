import os
import datetime
from serpapi import GoogleSearch

IATA_CODES = {
    "Karachi": "KHI",
    "Lahore": "LHE",
    "Islamabad": "ISB",
    "Peshawar": "PEW",
    "Quetta": "UET",
    "Multan": "MUX",
    "Sialkot": "SKT",
    "Faisalabad": "LYP",
    "Bahawalpur": "BHV",
    "Bangalore": "BLR",
    "Chennai": "MAA",
    "Delhi": "DEL",
    "Hyderabad": "HYD",
    "Kolkata": "CCU",
    "Mumbai": "BOM"
}

def get_live_flight_price(source_city, destination_city, airline, country):
    """
    Fetches real-time flight pricing using SerpApi's Google Flights Engine.
    Returns a dictionary with status, price, and source snippet.
    """
    currency = "PKR" if country == "Pakistan" else "INR"
    
    departure_id = IATA_CODES.get(source_city, source_city)
    arrival_id = IATA_CODES.get(destination_city, destination_city)
    
    # We will search for a flight tomorrow for a generic live price
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    outbound_date = tomorrow.strftime("%Y-%m-%d")

    # Get API key from environment variable or use placeholder
    api_key = os.environ.get("SERPAPI_KEY", "85a42d812e19917021faf7ad0a9b82e07f88ec1d445b81a227af1a7ae36df178")
    
    params = {
      "engine": "google_flights",
      "departure_id": departure_id,
      "arrival_id": arrival_id,
      "outbound_date": outbound_date,
      "currency": currency,
      "hl": "en",
      "type": "2",
      "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return {
                "status": "error",
                "message": results["error"]
            }

        flights = results.get("best_flights", [])
        if not flights:
            flights = results.get("other_flights", [])
            
        if flights:
            target_airline = airline.lower()
            matched_flight = None
            
            for f in flights:
                flight_airlines = []
                if "flights" in f:
                    for seg in f["flights"]:
                        if "airline" in seg:
                            flight_airlines.append(seg["airline"].lower())
                            
                # Check if the requested airline is in this itinerary
                if any(target_airline in a or a in target_airline for a in flight_airlines):
                    matched_flight = f
                    break
                    
            if not matched_flight:
                # If requested airline not found, just take the first one (cheapest)
                matched_flight = flights[0]
                
            price = matched_flight.get("price", 0)
            
            if price > 0:
                # Extract some snippet info
                airline_names = []
                if "flights" in matched_flight:
                    airline_names = [seg.get("airline", "") for seg in matched_flight["flights"]]
                airlines_str = ", ".join(set(airline_names))
                
                snippet = f"Google Flights: {airlines_str} - Departure: {outbound_date}"
                return {
                    "status": "success",
                    "price": price,
                    "currency": currency,
                    "snippet": snippet
                }
            
        return {
            "status": "not_found",
            "message": "Could not find any flights for this route on Google Flights."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    print(get_live_flight_price("Karachi", "Islamabad", "PIA", "Pakistan"))
    print(get_live_flight_price("Delhi", "Mumbai", "Vistara", "India"))
