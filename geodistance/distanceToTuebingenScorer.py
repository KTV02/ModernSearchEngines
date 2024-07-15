import sqlite3
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from src.gazetteer import load_gazetteer, generate_descriptor_names

def calculate_distance_to_tuebingen(lat, lon):
    tuebingen_coords = (48.521636, 9.057645)
    return geodesic(tuebingen_coords, (lat, lon)).kilometers

def filter_gazetteer_by_germany_places(df):
    return df.loc[
        (df['country_code'] == 'DE') &
        (df['feature_class'] == 'P')
    ]

def tag_gazetteer(html, gazetteer_df):
    matches = []
    for _, row in gazetteer_df.iterrows():
        name = row['name']
        if name in html:
            match = row.to_dict()
            matches.append(match)
    return matches

def process_website_content(content, gazetteer_df):
    matches = tag_gazetteer(content, gazetteer_df)
    distances = []
    geolocator = Nominatim(user_agent="geoapiExercises")
    for match in matches:
        lat, lon = match['latitude'], match['longitude']
        distance = calculate_distance_to_tuebingen(lat, lon)
        distances.append(distance)
    if distances:
        return np.mean(distances)
    return None

def main():
    # Load gazetteer data
    GAZETTEER_PATH = 'data/geonames/allCountries.txt'
    ADMIN1_PATH = 'data/geonames/admin1CodesASCII.txt'
    ADMIN2_PATH = 'data/geonames/admin2Codes.txt'
    COUNTRY_PATH = 'data/geonames/countryInfo.txt'
    FEATURE_PATH = 'data/geonames/featureCodes_en.txt'
    gazetteer_df = load_gazetteer(GAZETTEER_PATH)
    with_names_df = generate_descriptor_names(gazetteer_df, ADMIN1_PATH, ADMIN2_PATH, COUNTRY_PATH, FEATURE_PATH)
    gazetteer_df = filter_gazetteer_by_germany_places(with_names_df)
    
    # Connect to the SQLite database
    conn = sqlite3.connect('index.db')
    cursor = conn.cursor()

    # Fetch website contents
    cursor.execute("SELECT url, content FROM documents")
    rows = cursor.fetchall()

    scores = []

    for row in rows:
        doc_id, content = row
        avg_distance = process_website_content(content, gazetteer_df)
        if avg_distance is not None:
            scores.append((doc_id, avg_distance))
            print(doc_id,avg_distance)
    
    # Output the scores
    for doc_id, score in scores:
        print(f"Website ID: {doc_id}, Average Distance to Tübingen: {score:.2f} km")

    # Optional: Store the scores back in the database
    cursor.executemany("UPDATE documents SET avg_distance_to_tuebingen = ? WHERE id = ?", scores)
    conn.commit()
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
