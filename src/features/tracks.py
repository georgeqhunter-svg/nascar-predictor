"""Track reference table for the Next Gen era.

Every Cup track that has hosted a points race from 2022 onward. Track type is the
coarse 5-way bucket used by the baseline model; a finer embedding will live in
`track_similarity.py` once we have enough race data to learn one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrackType = Literal["superspeedway", "intermediate", "short", "road"]


@dataclass(frozen=True)
class Track:
    slug: str            # racing-reference /tracks/<slug>
    display: str         # human name
    length_mi: float
    surface: str         # 'oval', 'road', 'street', 'dirt'
    banking_deg: float | None
    track_type: TrackType
    lat: float           # for weather lookup
    lon: float

    @property
    def is_drafting(self) -> bool:
        return self.track_type == "superspeedway"


# The universe of Cup tracks in the Next Gen era. Update when a new venue is added.
# Values sourced from racing-reference track pages and NASCAR track specs.
# Coordinates are the track center (approx.), used only for weather station lookup.
TRACKS: dict[str, Track] = {
    t.slug: t for t in [
        Track("Daytona_International_Speedway", "Daytona", 2.500, "oval", 31.0, "superspeedway", 29.185, -81.070),
        Track("Talladega_Superspeedway", "Talladega", 2.660, "oval", 33.0, "superspeedway", 33.567, -86.066),
        Track("Atlanta_Motor_Speedway", "Atlanta", 1.540, "oval", 28.0, "superspeedway", 33.386, -84.316),

        Track("Las_Vegas_Motor_Speedway", "Las Vegas", 1.500, "oval", 20.0, "intermediate", 36.272, -115.010),
        Track("Kansas_Speedway", "Kansas", 1.500, "oval", 20.0, "intermediate", 39.115, -94.832),
        Track("Charlotte_Motor_Speedway", "Charlotte", 1.500, "oval", 24.0, "intermediate", 35.352, -80.683),
        Track("Homestead-Miami_Speedway", "Homestead", 1.500, "oval", 20.0, "intermediate", 25.452, -80.409),
        Track("Texas_Motor_Speedway", "Fort Worth", 1.500, "oval", 20.0, "intermediate", 33.037, -97.283),
        Track("Michigan_International_Speedway", "Michigan", 2.000, "oval", 18.0, "intermediate", 42.066, -84.242),
        Track("Darlington_Raceway", "Darlington", 1.366, "oval", 25.0, "intermediate", 34.294, -79.906),
        Track("Nashville_Superspeedway", "Nashville", 1.333, "oval", 14.0, "intermediate", 36.017, -86.408),
        Track("World_Wide_Technology_Raceway_at_Gateway", "Gateway", 1.250, "oval", 11.0, "intermediate", 38.652, -90.135),
        # Alias: races.parquet names this track "World Wide Technology Raceway"
        # (no "at Gateway"), so name-matching missed it and Gateway races got
        # NaN length/banking. Same physical track.
        Track("World_Wide_Technology_Raceway", "Gateway", 1.250, "oval", 11.0, "intermediate", 38.652, -90.135),

        Track("Martinsville_Speedway", "Martinsville", 0.526, "oval", 12.0, "short", 36.634, -79.851),
        Track("Bristol_Motor_Speedway", "Bristol", 0.533, "oval", 30.0, "short", 36.516, -82.257),
        Track("Richmond_Raceway", "Richmond", 0.750, "oval", 14.0, "short", 37.593, -77.418),
        Track("Phoenix_Raceway", "Phoenix", 1.000, "oval", 11.0, "short", 33.375, -112.310),
        Track("New_Hampshire_Motor_Speedway", "Loudon", 1.058, "oval", 7.0, "short", 43.363, -71.461),
        Track("Dover_Motor_Speedway", "Dover", 1.000, "oval", 24.0, "short", 39.190, -75.531),
        Track("Iowa_Speedway", "Iowa", 0.875, "oval", 14.0, "short", 41.616, -93.122),

        Track("Sonoma_Raceway", "Sonoma", 1.990, "road", None, "road", 38.161, -122.454),
        Track("Watkins_Glen_International", "Watkins Glen", 2.450, "road", None, "road", 42.338, -76.923),
        Track("Circuit_of_the_Americas", "Austin", 3.410, "road", None, "road", 30.135, -97.641),
        Track("Chicago_Street_Course", "Chicago", 2.200, "street", None, "road", 41.876, -87.622),
        Track("Charlotte_Motor_Speedway_Road_Course", "Roval", 2.320, "road", None, "road", 35.352, -80.683),
        Track("Indianapolis_Motor_Speedway_Road_Course", "Indy RC", 2.439, "road", None, "road", 39.795, -86.234),
        Track("Autodromo_Hermanos_Rodriguez", "Mexico City", 2.674, "road", None, "road", 19.404, -99.089),

        # Pocono + Indy — reclassified from "unique" to intermediate. Both race
        # with intermediate engine packages, aero-dependent, fuel-window pit
        # strategy, no pack racing. Track length + banking differences are
        # captured by the model's continuous features (track_length_mi,
        # track_banking_deg) so they don't need a separate type bucket.
        Track("Pocono_Raceway", "Pocono", 2.500, "oval", 14.0, "intermediate", 41.055, -75.512),
        Track("Indianapolis_Motor_Speedway", "Indianapolis", 2.500, "oval", 9.0, "intermediate", 39.795, -86.234),

        Track("North_Wilkesboro_Speedway", "North Wilkesboro", 0.625, "oval", 14.0, "short", 36.132, -81.055),
        Track("Chicago_Street_Race", "Chicago Street", 2.200, "street", None, "road", 41.876, -87.622),
        Track("Auto_Club_Speedway", "Auto Club", 2.000, "oval", 14.0, "intermediate", 34.089, -117.501),
        Track("Road_America", "Road America", 4.048, "road", None, "road", 43.798, -87.995),
        # Bristol Dirt was a one-off dirt experiment 2021-2023 — retired. Kept
        # so historical races validate; classified as "short" (closer to short-
        # track racing dynamics than intermediate) as part of scrapping the
        # "unique" category.
        Track("Bristol_Motor_Speedway_Dirt", "Bristol Dirt", 0.533, "dirt", 30.0, "short", 36.516, -82.257),
        Track("Chicagoland_Speedway", "Chicagoland", 1.500, "oval", 18.0, "intermediate", 41.475, -88.058),
        Track("San_Diego_Street_Course", "San Diego Street", 2.100, "street", None, "road", 32.716, -117.161),
    ]
}




def get_track(slug: str) -> Track | None:
    """Look up a track by racing-reference slug."""
    return TRACKS.get(slug)


def resolve_track_type(track_name: str, fallback: str = "intermediate") -> str:
    """Return the authoritative track_type for a race, per tracks.py.

    Parquet-stored track_type values can go stale after a reclassification
    (e.g., Pocono/Indy moved from "unique" to "intermediate"). This helper
    matches by track_name against the TRACKS table so all callers stay in
    sync with the current classification.
    """
    slug_key = str(track_name).lower()
    for slug, tobj in TRACKS.items():
        if slug.replace("_", " ").lower() == slug_key:
            return tobj.track_type
    return fallback


def classify_by_slug(slug: str) -> TrackType:
    """Return the track type, defaulting to 'intermediate' for unknown ovals."""
    t = TRACKS.get(slug)
    return t.track_type if t else "intermediate"
