"""Member 2 features; drop-in replacement for the supplied features.py.

No labels, sale dates, AVM estimates, or fitted target aggregates are accessed.
Native pandas categories require scikit-learn >= 1.6 and the starter model's
categorical_features='from_dtype' (the default in the pinned 1.7.2 version).
Frozen category vocabularies below come only from sales before December 2025.
Unknown categories become missing, with no change to the column schema.
"""
import re

import numpy as np
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]

# CATEGORY_LEVELS is populated below from the pre-validation training period.
CATEGORY_LEVELS = {'municipality': ['Etobicoke', 'North York', 'Scarborough', 'Toronto', 'York'],
 'community': ['Alderhaven',
               'Alderworth',
               'Ashbank',
               'Ashhollow',
               'Ashpark',
               'Ashworth',
               'Aspenwood',
               'Aspenworth',
               'Balsamcrest',
               'Balsamgrove',
               'Balsamwood',
               'Beechbourne',
               'Beechcrest',
               'Beechglen',
               'Birchbrook',
               'Birchgrove',
               'Birchheights',
               'Birchhollow',
               'Cedarbank',
               'Cedargardens',
               'Cedarglen',
               'Cedargrove',
               'Cedarhaven',
               'Cedarheights',
               'Cedarview',
               'Cedarwood',
               'Chestnutbank',
               'Chestnutbourne',
               'Chestnutbrook',
               'Chestnutglen',
               'Chestnutgrove',
               'Chestnutmills',
               'Chestnutridge',
               'Chestnutvale',
               'Chestnutview',
               'Cypressbrook',
               'Cypressdale',
               'Cypressglen',
               'Cypressheights',
               'Cypresspark',
               'Dogwoodbrook',
               'Dogwoodcliffe',
               'Dogwoodglen',
               'Dogwoodhaven',
               'Dogwoodmills',
               'Elmbank',
               'Elmbourne',
               'Elmbrook',
               'Elmcrest',
               'Elmgardens',
               'Elmhaven',
               'Elmmills',
               'Elmpark',
               'Elmview',
               'Hawthornbourne',
               'Hawthorncliffe',
               'Hazelbourne',
               'Hazelfield',
               'Hazelmills',
               'Hemlockhollow',
               'Hemlockvale',
               'Hickorybourne',
               'Hickoryglen',
               'Hickoryhaven',
               'Hickorymills',
               'Hickoryridge',
               'Hickoryvale',
               'Juniperbrook',
               'Junipergardens',
               'Juniperhaven',
               'Juniperheights',
               'Juniperpark',
               'Larchbank',
               'Larchview',
               'Lindencliffe',
               'Lindencrest',
               'Lindendale',
               'Lindenglen',
               'Lindenmills',
               'Lindenridge',
               'Lindenvale',
               'Magnoliabank',
               'Magnoliacrest',
               'Magnoliaglen',
               'Magnoliavale',
               'Magnoliaview',
               'Magnoliaworth',
               'Maplebank',
               'Maplebourne',
               'Maplebrook',
               'Maplegrove',
               'Maplemills',
               'Maplepark',
               'Maplevale',
               'Mapleview',
               'Mulberrydale',
               'Mulberrygrove',
               'Mulberryhaven',
               'Mulberryhollow',
               'Mulberryvale',
               'Oakbourne',
               'Oakcrest',
               'Oakhaven',
               'Oakview',
               'Pineglen',
               'Pineridge',
               'Pinevale',
               'Poplarcrest',
               'Poplargardens',
               'Poplarview',
               'Rowanbank',
               'Rowanbrook',
               'Rowangardens',
               'Rowangrove',
               'Rowanhollow',
               'Rowanridge',
               'Sequoiabourne',
               'Sequoiafield',
               'Sequoiahollow',
               'Sequoiaview',
               'Sprucehaven',
               'Sprucehollow',
               'Spruceview',
               'Sumacbank',
               'Sumaccrest',
               'Sumacdale',
               'Sumacfield',
               'Sumacglen',
               'Sumacpark',
               'Sumacvale',
               'Sumacwood',
               'Sumacworth',
               'Sycamoredale',
               'Sycamoreglen',
               'Sycamorehaven',
               'Sycamoreview',
               'Sycamorewood',
               'Tamarackbourne',
               'Tamarackmills',
               'Walnutfield',
               'Walnuthollow',
               'Walnutpark',
               'Willowglen',
               'Willowpark',
               'Willowview'],
 'house_style': ['1 1/2 Storey',
                 '1 Storey/Apt',
                 '2 1/2 Storey',
                 '2-Storey',
                 '3-Storey',
                 'Apartment',
                 'Backsplit 3',
                 'Backsplit 4',
                 'Backsplit 5',
                 'Bungalow',
                 'Bungalow-Raised',
                 'Loft',
                 'Multi-Level',
                 'Sidesplit 3',
                 'Sidesplit 4',
                 'Stacked Townhse']}

TEXT_PATTERNS = {
    "turnkey": r"\b(?:turnkey|move.in.ready|fully redone|completely redone|nothing to do but|top to bottom)\b",
    "renovated": r"\b(?:renovated|upgraded|remodelled|remodeled|updated kitchen|kitchen (?:has been )?(?:updated|redone)|new kitchen)\b",
    "cosmetic_update": r"\b(?:fresh(?:ly)? paint(?:ed)?|newer flooring|new countertops|refreshed bathroom|modern light fixtures)\b",
    "needs_work": r"\b(?:as.is|needs? (?:work|updating|paint|replacing)|requires? (?:updating|renovation)|dated|tired|bring your contractor|fixer|cosmetic overhaul)\b",
    "original_finishes": r"\b(?:original (?:finishes|kitchen|bathroom|light fixtures)|older (?:appliances|windows)|good bones|great bones|modernize|modernise)\b",
    "luxury_finishes": r"\b(?:luxury|luxurious|spa.like|designer|custom (?:cabinetry|closet)|quartz|marble|heated (?:bathroom )?floors|wide.plank)\b",
    "finished_basement": r"(?<!un)\bfinished basement\b|\brenovated basement\b",
    "unfinished_basement": r"\bunfinished basement\b",
    "separate_entrance": r"\b(?:separate (?:side )?entrance|walk.out basement|basement apartment|in.law suite)\b",
    "den": r"\b(?:den|home office)\b",
    "view": r"\b(?:unobstructed view|lake view|waterfront|ravine.backing)\b",
    "outdoor_space": r"\b(?:balcony|terrace|private patio|private backyard|fenced yard|deck)\b",
    "transit": r"\b(?:transit|subway|ttc|go station)\b",
    "amenities": r"\b(?:concierge|gym|pool|guest suites|party room)\b",
    "motivated_seller": r"\b(?:motivated seller|seller is motivated|priced to sell)\b",
}


def square_feet(house_area):
    """Parse numbers or ranges, including comma-separated sizes and unit suffixes.

    Open-ended intervals use their stated bound; separate indicators let the
    model distinguish these less precise measurements.
    """
    text = house_area.astype("string").str.replace(",", "", regex=False)
    text = text.str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
    parts = text.str.extract(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?")
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return ((low + high) / 2).where(low > 0).astype(float)


def _number(homes, column):
    return pd.to_numeric(homes[column], errors="coerce").astype(float)


def _ratio(numerator, denominator):
    return numerator.div(denominator.where(denominator > 0))


def structured_features(homes):
    """Property, location and calendar features available when listed."""
    cols = {}
    area = square_feet(homes["house_area"])
    cols["square_feet"] = area
    for source, name in [("bedrooms", "bedrooms"), ("washrooms", "washrooms"),
                         ("parking_total", "parking"), ("garage", "garage"),
                         ("latitude", "latitude"), ("longitude", "longitude"),
                         ("lot_front", "lot_front"), ("lot_depth", "lot_depth")]:
        cols[name] = _number(homes, source)
    lot = cols["lot_front"].where(cols["lot_front"] > 0) * cols["lot_depth"].where(cols["lot_depth"] > 0)
    cols["lot_area"] = lot
    cols["log_square_feet"] = np.log1p(area)
    cols["log_lot_area"] = np.log1p(lot)
    cols["lot_to_floor_area"] = _ratio(lot, area)
    cols["lot_aspect_ratio"] = _ratio(cols["lot_depth"], cols["lot_front"])
    cols["sqft_per_bedroom"] = _ratio(area, cols["bedrooms"])
    cols["sqft_per_washroom"] = _ratio(area, cols["washrooms"])
    cols["washrooms_per_bedroom"] = _ratio(cols["washrooms"], cols["bedrooms"])
    cols["non_garage_parking"] = (cols["parking"] - cols["garage"]).clip(lower=0)
    cols["area_missing"] = area.isna().astype(int)
    cols["lot_missing"] = lot.isna().astype(int)
    area_text = homes["house_area"].astype("string")
    cols["area_is_range"] = area_text.str.contains(r"\d\s*[-–—]\s*\d", na=False).astype(int)
    cols["area_open_bound"] = area_text.str.contains(r"[<>+]|over|under", case=False, na=False).astype(int)
    date = pd.to_datetime(homes["date_listed"], errors="coerce")
    cols["months"] = (date - pd.Timestamp("2021-01-01")).dt.days / 30.44
    cols["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    cols["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    # Retain the original type flags so existing tuned code can still select them.
    for kind in PROPERTY_TYPES:
        cols["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    X = pd.DataFrame(cols, index=homes.index)
    for column, levels in CATEGORY_LEVELS.items():
        X[column] = pd.Categorical(homes[column], categories=levels)
    return X


def off_market_features(homes):
    """Structured + interpretable text; deliberately never reads list_price."""
    X = structured_features(homes)
    text = homes["remarks"].fillna("").astype(str).str.lower()
    for name, pattern in TEXT_PATTERNS.items():
        X["text_" + name] = text.str.contains(pattern, regex=True, flags=re.I).astype(int)
    X["remarks_missing"] = text.str.strip().eq("").astype(int)
    return X


def list_aware_features(homes):
    """Same features, plus listing-price magnitude and price-per-unit context."""
    X = off_market_features(homes)
    price = _number(homes, "list_price").where(lambda s: s > 0)
    X["list_price"] = price
    X["log_list_price"] = np.log(price)
    X["list_price_per_sqft"] = _ratio(price, X["square_feet"])
    X["list_price_per_lot_sqft"] = _ratio(price, X["lot_area"])
    return X
