"""Modul for å bygge et dataset"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from poweranalysis.weather.frost_client import FrostDataRetriever
from poweranalysis.config import Config


def _ensure_directories(config: Config) -> None:
    """Sørger for at nødvendige mapper finnes"""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.external_dir.mkdir(parents=True, exist_ok=True)


def _normalize_timezone(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    Policy: Sørger for DatetimeIndex, sortering, dedupe.
    Hvis tz-aware -> konverter til lokal tid og gjør tz-naiv for konsistent merge.
    """
    out = df.copy()

    if "Tidspunkt" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["Tidspunkt"] = pd.to_datetime(out["Tidspunkt"], errors="coerce")
        out = out.dropna(subset=["Tidspunkt"]).set_index("Tidspunkt")

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert(tz).tz_localize(None)

    out = out[~out.index.duplicated(keep="last")]
    return out


def load_power(config: Config) -> pd.DataFrame:
    """Laster inn og standardiserer forbruksdata."""
    frames: list[pd.DataFrame] = []

    for file in config.power_files:
        path = config.power_dir / file
        df = pd.read_excel(path, skiprows=config.skiprows).dropna(how="all").copy()

        # Standardiser navn
        if "Nettleie* (øre/kWh)" in df.columns:
            df = df.rename(columns={"Nettleie* (øre/kWh)": "Nettleie (øre/kWh)"})

        df = _normalize_timezone(df, config.timezone)
        frames.append(df)

    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _load_or_fetch_weather(
    config: Config,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force_fetch: bool = False,
) -> pd.DataFrame:
    """Laster inn værdata, eller henter fra API hvis ikke cache finnes."""
    if config.weather_cache.exists() and not force_fetch:
        df = pd.read_parquet(config.weather_cache)
        return _normalize_timezone(df, config.timezone)

    client_id = os.getenv("FROST_CLIENT_ID")
    client_secret = os.getenv("FROST_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Mangler Frost API credentials. Sett miljøvariablene.")

    with FrostDataRetriever(client_id, client_secret) as frost:
        df_weather, _ = frost.get_observations(
            lat=config.lat,
            lon=config.lon,
            start=start,
            end=end,
            elements=list(config.wanted_elements),
            max_stations=config.max_stations,
            output="df_fallback",
        )

    df_weather = _normalize_timezone(df_weather, config.timezone)
    df_weather.to_parquet(config.weather_cache)
    return df_weather


def _merge_power_weather(df_power: pd.DataFrame, df_weather: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Slår sammen forbruksdata og værdata."""
    df_power = _normalize_timezone(df_power, config.timezone).sort_index()
    df_weather = _normalize_timezone(df_weather, config.timezone).sort_index()

    df_merged = pd.merge_asof(
        df_power,
        df_weather,
        left_index=True,
        right_index=True,
        direction=config.join_direction,
        tolerance=pd.Timedelta(config.join_tolerance),
    )

    if "air_temperature" in df_merged.columns:
        df_merged["air_temperature"] = df_merged["air_temperature"].interpolate(method="time")

    return df_merged


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legger til grunnleggende features."""
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = out["dayofweek"] >= 5
    out["month"] = out.index.month
    return out


def _make_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregerer til daglig nivå."""
    daily = df.resample("D").mean(numeric_only=True)

    col = "Forbruk (kWh)"
    if col in df.columns:
        daily[col] = df[col].resample("D").sum(min_count=1)

    return daily


def build_dataset(config: Config, force_fetch_weather: bool = False) -> None:
    """Hovedfunksjon for å bygge dataset."""
    _ensure_directories(config)

    df_power = load_power(config)
    start = pd.Timestamp(df_power.index.min())
    end = pd.Timestamp(df_power.index.max())

    df_weather = _load_or_fetch_weather(config, start, end, force_fetch=force_fetch_weather)
    df_merged = _merge_power_weather(df_power, df_weather, config)

    df_final = _add_basic_features(df_merged)
    df_daily = _make_daily(df_final)

    df_final.to_parquet(config.out_hourly)
    df_daily.to_parquet(config.out_daily)


def main() -> None:
    config = Config()
    build_dataset(config)
    print("Dataset bygging fullført.")


if __name__ == "__main__":
    main()