from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    #  Paths 
    project_root: Path = PROJECT_ROOT

    power_dir = project_root / "data" / "raw" / "power"
    temperature_dir = project_root / "data" / "raw" / "temperatur"

    power_files: Tuple = (
        "Aug.xls",
        "Sep.xls",
        "Okt.xls",
        "Nov.xls",
        "des.xls",
    )
    skiprows: int = 6

    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    external_dir: Path = PROJECT_ROOT / "data" / "external"
    weather_cache: Path = external_dir / "weather.parquet"
    out_hourly: Path = processed_dir / "dataset_hourly.parquet"
    out_daily: Path = processed_dir / "dataset_daily.parquet"

    ## Vær og plassering
    lat: float = 63.4305
    lon: float = 10.3951
    max_stations: int = 20
    wanted_elements: Tuple = (
        "air_temperature",
        "relative_humidity",
        "wind_speed",
        "wind_from_direction",
        "sum(precipitation_amount PT1H)",
    )

    # Join
    join_tolerance: str = "1H"
    join_direction: str = "nearest"

    # Tidsone
    timezone: str = "Europe/Oslo"
