"""Canonical converters from raw tabular flight/aircraft data into the FLIGHTS/PLANES shapes
AirlineEnv expects. Shared by both real ROADEF instance loading (roadef.py) and synthetic
instance generation (synthetic.py), so every instance source produces identically-shaped output.
"""

import random

import pandas as pd


def build_flight_pool(flights_df: pd.DataFrame, itineraries_df: pd.DataFrame, empty_flights: bool = True):
    merged = flights_df.merge(itineraries_df, on="flight_id", how="left")
    merged["total_ticket_price"] = merged["total_ticket_price"].fillna(0.0)
    merged["total_passenger_count"] = merged["total_passenger_count"].fillna(0)

    if not empty_flights:
        merged = merged[merged.total_ticket_price > 0]

    pool = []
    for row in merged.to_dict("records"):
        passenger_count = int(max(1, round(float(row["total_passenger_count"]))))
        pool.append(
            {
                "id": int(row["flight_id"]),
                "origin": str(row["origin"]).strip().upper(),
                "dest": str(row["destination"]).strip().upper(),
                "start": int(row["start_min"]),
                "pass": passenger_count,
                "total_ticket_price": float(row["total_ticket_price"]),
            }
        )

    return pool


def build_planes(aircraft_df: pd.DataFrame):
    planes = {}
    for row in aircraft_df.to_dict("records"):
        plane_id = str(row["aircraft_id"])
        planes[plane_id] = {
            "fixed_cost": float(row["fixed_cost"]),
            "hourly_cost": float(row["hourly_cost"]),
            "initial_airport": str(row["initial_airport"]).strip().upper(),
            "seats": int(row["seats"]),
            "speed": float(row["speed"]),
        }
    return planes


"""Builds the concrete schedule an experiment trains on, from a loaded ROADEF instance."""

from roadef import Instance
from synthetic import synthetic_schedule_like

SCHEDULE_TYPES = ("sample", "random", "trap")


def build_schedule(
    base: Instance,
    schedule_type: str,
    max_flights: int | None,
    max_planes: int | None,
    n_cities: int,
    seed: int,
) -> Instance:
    """Returns (flights, planes, airports, dist_dict) for one experiment schedule.

    - "sample": max_flights real flights sampled from the base instance
    - "random" / "trap": max_flights synthetic flights between n_cities of the base instance's
      airports, priced like it (see synthetic_schedule_like)

    The fleet is always max_planes aircraft sampled from the base instance, and dist_dict is the
    base instance's (a superset of the airports used). None for max_* keeps the full count.
    """
    flights, planes, _, dist_dict = base
    if schedule_type == "sample":
        flights, planes, airports = subsample_instance(
            flights, planes, max_flights=max_flights, max_planes=max_planes, seed=seed
        )
    elif schedule_type in ("random", "trap"):
        n_flights = max_flights if max_flights is not None else len(flights)
        synthetic = synthetic_schedule_like(flights, schedule_type, n_flights, n_cities)
        flights, planes, airports = subsample_instance(
            synthetic, planes, max_flights=None, max_planes=max_planes, seed=seed
        )
    else:
        raise ValueError(f"Unknown schedule type {schedule_type!r}; expected one of {SCHEDULE_TYPES}")
    return flights, planes, airports, dist_dict

def subsample_instance(
    flights: list[dict],
    planes: dict[str, dict],
    max_flights: int | None = None,
    max_planes: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict], dict[str, dict], list[str]]:
    """Randomly downsizes a loaded instance's flights/planes for faster iteration, and recomputes
    the airport list to match. Pass max_flights/max_planes=None to leave that side untouched.

    The instance's dist_dict does not need rebuilding after this: it stays a (harmless) superset
    of the airports actually used by the returned flights/planes.
    """
    rng = random.Random(seed)

    if max_planes is not None and max_planes < len(planes):
        kept_ids = rng.sample(list(planes), max_planes)
        planes = {plane_id: planes[plane_id] for plane_id in kept_ids}

    if max_flights is not None and max_flights < len(flights):
        flights = rng.sample(flights, max_flights)
    flights = sorted(flights, key=lambda f: f["start"])

    airports = sorted(
        {f["origin"] for f in flights} | {f["dest"] for f in flights} | {p["initial_airport"] for p in planes.values()}
    )
    return flights, planes, airports