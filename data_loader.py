"""
data_loader.py — Load, clean, and enrich the Nassau Candy Distributor dataset.
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# Factory definitions
# ──────────────────────────────────────────────
FACTORY_COORDS = {
    "Lot's O' Nuts":    (32.881893, -111.768036),
    "Wicked Choccy's":  (32.076176,  -81.088371),
    "Sugar Shack":      (48.119140,  -96.181150),
    "Secret Factory":   (41.446333,  -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# US state centres (approximate) for map plotting
US_STATE_COORDS = {
    "Alabama": (32.806671, -86.791130),
    "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954456),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    # Canadian provinces
    "Ontario": (51.253775, -85.323214),
    "Quebec": (52.939916, -73.549136),
    "British Columbia": (53.726669, -127.647621),
    "Alberta": (53.933271, -116.576503),
    "Nova Scotia": (44.681987, -63.744311),
}


def load_and_clean(csv_path: str | None = None) -> pd.DataFrame:
    """Load the CSV, parse dates, clean, and enrich with factory/route data."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "Nassau Candy Distributor.csv")

    df = pd.read_csv(csv_path)

    # Parse dates (DD-MM-YYYY)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

    # Drop rows with missing dates
    df.dropna(subset=["Order Date", "Ship Date"], inplace=True)

    # Standardize text fields
    for col in ["City", "State/Province", "Country/Region", "Region", "Division", "Product Name"]:
        df[col] = df[col].astype(str).str.strip()

    # Shipping lead time (days)
    df["Shipping_Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Remove invalid / negative lead times
    df = df[df["Shipping_Lead_Time"] >= 0].copy()

    # Map product → factory
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP).fillna("Unknown")

    # Factory coordinates
    df["Factory_Lat"] = df["Factory"].map(lambda f: FACTORY_COORDS.get(f, (None, None))[0])
    df["Factory_Lon"] = df["Factory"].map(lambda f: FACTORY_COORDS.get(f, (None, None))[1])

    # Customer state coordinates
    df["State_Lat"] = df["State/Province"].map(lambda s: US_STATE_COORDS.get(s, (None, None))[0])
    df["State_Lon"] = df["State/Province"].map(lambda s: US_STATE_COORDS.get(s, (None, None))[1])

    # Route label
    df["Route"] = df["Factory"] + " → " + df["State/Province"]

    # Route (region level)
    df["Route_Region"] = df["Factory"] + " → " + df["Region"]

    return df
