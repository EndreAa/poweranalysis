"""Client library for retrieving weather observations from the Frost API.

This module provides a small, production-style client used by notebooks in this
repository. It focuses on:
- explicit typing
- input validation
- robust HTTP error handling
- predictable logging behavior
- backward compatibility with existing notebook code
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ElementInput = Optional[Union[str, Iterable[str]]]


class FrostAPIError(RuntimeError):
    """Raised when the Frost API returns an error or an invalid response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        reason: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.reason = reason
        self.endpoint = endpoint
        super().__init__(self.__str__())

    def __str__(self) -> str:
        base = f"Frost API error [{self.status_code}] {self.message}"
        if self.reason:
            base += f" (reason: {self.reason})"
        if self.endpoint:
            base += f" @ {self.endpoint}"
        return base


@dataclass(frozen=True)
class FrostAPIConfig:
    """Configuration for Frost API communication."""

    base_url: str = "https://frost.met.no"
    observations_endpoint: str = "/observations/v0.jsonld"
    sources_endpoint: str = "/sources/v0.jsonld"
    timeseries_endpoint: str = "/observations/availableTimeSeries/v0.jsonld"
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.5
    user_agent: str = "stromanalyse-frost-client/1.0"


class FrostDataRetriever:
    """Typed Frost API client.

    Parameters
    ----------
    client_id:
        Frost API client ID.
    client_secret:
        Frost API client secret.
    config:
        Optional API configuration.
    session:
        Optional preconfigured requests session, useful for tests.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str = "",
        config: Optional[FrostAPIConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not client_id or not client_id.strip():
            raise ValueError("client_id must be a non-empty string.")

        self.client_id = client_id.strip()
        self.client_secret = client_secret
        self.config = config or FrostAPIConfig()
        self._session = session or self._create_session()
        logger.info("Initialized FrostDataRetriever.")

    @property
    def endpoint_observations(self) -> str:
        return f"{self.config.base_url}{self.config.observations_endpoint}"

    @property
    def endpoint_sources(self) -> str:
        return f"{self.config.base_url}{self.config.sources_endpoint}"

    @property
    def endpoint_timeseries(self) -> str:
        return f"{self.config.base_url}{self.config.timeseries_endpoint}"

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "FrostDataRetriever":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        logger.info("Closed FrostDataRetriever session.")

    def get_stations(self, lat: float, lon: float, max_count: int = 5) -> List[Dict[str, Any]]:
        """Return nearest weather stations for a coordinate."""
        self._validate_lat_lon(lat, lon)
        if max_count < 1:
            raise ValueError("max_count must be >= 1.")

        params = {
            "geometry": f"nearest(POINT({lon} {lat}))",
            "nearestmaxcount": max_count,
        }
        return self._request(self.endpoint_sources, params)

    def get_available_period(self, station: str) -> List[Dict[str, Any]]:
        """Return available time series metadata for a station."""
        if not station or not station.strip():
            raise ValueError("station must be a non-empty string.")
        return self._request(self.endpoint_timeseries, {"sources": station.strip()})

    def get_observations(
        self,
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        sources: Optional[str] = None,
        elements: ElementInput = None,
        *,
        output: Literal["raw", "df", "df_fallback"] = "raw",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        max_stations: int = 20,
        include_wind_filters: bool = True,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> Union[List[Dict[str, Any]], pd.DataFrame, Tuple[pd.DataFrame, Dict[str, str]]]:
        """Fetch observations in one unified interface.

        Parameters
        ----------
        output:
            - ``"raw"``: returns raw API payload (list of dicts)
            - ``"df"``: returns pivoted DataFrame
            - ``"df_fallback"``: returns ``(DataFrame, station_map)`` using nearest-station fallback
        """
        if output == "raw":
            if not sources or not sources.strip():
                raise ValueError("sources must be provided when output='raw'.")
            return self._get_observations_raw(
                start=start,
                end=end,
                sources=sources,
                elements=elements,
                include_wind_filters=include_wind_filters,
                extra_params=extra_params,
            )

        if output == "df":
            if not sources or not sources.strip():
                raise ValueError("sources must be provided when output='df'.")
            return self.get_observations_df(
                start=start,
                end=end,
                sources=sources,
                elements=elements,
            )

        if output == "df_fallback":
            if lat is None or lon is None:
                raise ValueError("lat and lon must be provided when output='df_fallback'.")
            element_list = self._normalize_element_list(elements)
            return self.get_observations_df_with_station_fallback(
                start=start,
                end=end,
                lat=lat,
                lon=lon,
                elements=element_list,
                max_stations=max_stations,
            )

        raise ValueError(f"Unsupported output mode: {output}")

    def _get_observations_raw(
        self,
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        sources: str,
        elements: ElementInput = None,
        *,
        include_wind_filters: bool = True,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return raw observation payload from Frost API."""
        if not sources or not sources.strip():
            raise ValueError("sources must be a non-empty string.")

        start_s = self._normalize_reference_time(start)
        end_s = self._normalize_reference_time(end)
        elements_csv = self._normalize_elements(elements)

        params: Dict[str, Any] = {
            "sources": sources.strip(),
            "referencetime": f"{start_s}/{end_s}",
        }
        if elements_csv:
            params["elements"] = elements_csv

        if include_wind_filters and elements_csv and "wind_speed" in elements_csv:
            # Frost can return duplicate wind series without these filters.
            params["levels"] = "0,10"
            params["performancecategories"] = "A"

        if extra_params:
            params.update(dict(extra_params))

        try:
            return self._request(self.endpoint_observations, params)
        except FrostAPIError as exc:
            # 412 can happen with strict wind filters; retry once without them.
            if exc.status_code == 412 and ("levels" in params or "performancecategories" in params):
                logger.warning("Received 412 with wind filters. Retrying without level/performance filters.")
                params.pop("levels", None)
                params.pop("performancecategories", None)
                return self._request(self.endpoint_observations, params)
            raise

    def get_observations_df(
        self,
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        sources: str,
        elements: ElementInput = None,
    ) -> pd.DataFrame:
        """Return observations as a pivoted DataFrame indexed by referenceTime."""
        data = self._get_observations_raw(start=start, end=end, sources=sources, elements=elements)

        records: List[Dict[str, Any]] = []
        for row in data:
            ref_time = row.get("referenceTime")
            for obs in row.get("observations", []):
                records.append(
                    {
                        "referenceTime": ref_time,
                        "elementId": obs.get("elementId"),
                        "value": obs.get("value"),
                    }
                )

        if not records:
            logger.warning("No observations returned for sources=%s elements=%s", sources, elements)
            return pd.DataFrame()

        raw = pd.DataFrame(records)
        df = raw.pivot_table(index="referenceTime", columns="elementId", values="value", aggfunc="mean")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        return df

    def get_observations_df_with_station_fallback(
        self,
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        lat: float,
        lon: float,
        elements: List[str],
        max_stations: int = 20,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Fetch each element from the first nearby station that has data."""
        if not elements:
            raise ValueError("elements must contain at least one element ID.")

        self._validate_lat_lon(lat, lon)
        stations = self.get_stations(lat, lon, max_count=max_stations)
        stations_df = pd.DataFrame(stations)
        if stations_df.empty or "id" not in stations_df.columns:
            logger.warning("No stations found for fallback lookup.")
            return pd.DataFrame(), {}

        station_ids = stations_df["id"].dropna().astype(str).tolist()
        merged_parts: List[pd.DataFrame] = []
        used_station_per_element: Dict[str, str] = {}

        for element in elements:
            found = False
            for station_id in station_ids:
                try:
                    df_el = self.get_observations_df(
                        start=start,
                        end=end,
                        sources=station_id,
                        elements=[element],
                    )
                except FrostAPIError as exc:
                    logger.warning("Failed %s from %s: %s", element, station_id, exc)
                    continue

                if df_el.empty or element not in df_el.columns:
                    continue

                non_null = int(df_el[element].notna().sum())
                if non_null == 0:
                    continue

                part = df_el[[element]].copy()
                part[f"{element}__station_id"] = station_id
                merged_parts.append(part)
                used_station_per_element[element] = station_id
                logger.info("Using station %s for %s (%d rows).", station_id, element, non_null)
                found = True
                break

            if not found:
                logger.warning("No station had usable data for element: %s", element)

        if not merged_parts:
            return pd.DataFrame(), used_station_per_element

        return pd.concat(merged_parts, axis=1).sort_index(), used_station_per_element

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.auth = (self.client_id, self.client_secret)
        session.headers.update({"User-Agent": self.config.user_agent})

        retry = Retry(
            total=self.config.max_retries,
            connect=self.config.max_retries,
            read=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _request(self, endpoint: str, parameters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        try:
            response = self._session.get(endpoint, params=dict(parameters), timeout=self.config.timeout_seconds)
        except RequestException as exc:
            raise FrostAPIError(
                status_code=0,
                message=f"Network error: {exc}",
                endpoint=endpoint,
            ) from exc

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                raise FrostAPIError(
                    status_code=200,
                    message="Invalid JSON in successful response.",
                    endpoint=endpoint,
                ) from exc
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise FrostAPIError(
                    status_code=200,
                    message="Unexpected payload shape: 'data' is not a list.",
                    endpoint=endpoint,
                )
            return data

        message = f"HTTP {response.status_code}"
        reason: Optional[str] = None
        try:
            error_payload = response.json().get("error", {})
            message = str(error_payload.get("message", message))
            reason_raw = error_payload.get("reason")
            reason = str(reason_raw) if reason_raw is not None else None
        except ValueError:
            # Keep default message if payload is not JSON.
            pass

        raise FrostAPIError(
            status_code=response.status_code,
            message=message,
            reason=reason,
            endpoint=endpoint,
        )

    @staticmethod
    def _normalize_elements(elements: ElementInput) -> Optional[str]:
        if elements is None:
            return None
        if isinstance(elements, str):
            normalized = elements.strip()
            return normalized or None
        result = [str(el).strip() for el in elements if str(el).strip()]
        return ",".join(result) if result else None

    @staticmethod
    def _normalize_element_list(elements: ElementInput) -> List[str]:
        if elements is None:
            return []
        if isinstance(elements, str):
            parts = [part.strip() for part in elements.split(",")]
            return [part for part in parts if part]
        return [str(el).strip() for el in elements if str(el).strip()]

    @staticmethod
    def _normalize_reference_time(value: Union[str, datetime, pd.Timestamp]) -> str:
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError("start/end cannot be empty strings.")
            return candidate.split("+")[0]
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")
        raise TypeError("start/end must be str, datetime, or pandas.Timestamp.")

    @staticmethod
    def _validate_lat_lon(lat: float, lon: float) -> None:
        if not (-90.0 <= lat <= 90.0):
            raise ValueError("lat must be between -90 and 90.")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError("lon must be between -180 and 180.")



__all__ = [
    "FrostAPIConfig",
    "FrostAPIError",
    "FrostDataRetriever",

]
