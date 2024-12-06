#! python
import json
import os
from dataclasses import dataclass
from datetime import datetime
from functools import cache

import click
import plotly
import plotly.graph_objs as go
import polars as pl
import requests
from dotenv import load_dotenv

script_path = os.path.dirname(os.path.abspath(__file__))

API_BASE_URL = "https://api.eia.gov/v2"

ROOT_CONTEXT = ["electricity", "electric-power-operational-data"]


@dataclass
class LocationFacet:
    id: str
    name: str
    alias: str


class EIARestAPIClient:
    _access_token: str

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token

    def execute_api_request(
        self, context: list[str], parameters: dict | None = None
    ) -> dict:
        target_slug = "/".join(context)
        target_url = f"{API_BASE_URL}/{target_slug}?api_key={self._access_token}"

        response = requests.get(
            target_url,
            headers={"X-Params": json.dumps(parameters)} if parameters else None,
        )

        if not response.ok:
            raise Exception(
                f"Unexpected response: status - {
                    response.status_code} ({response.content})"
            )

        return response.json()


@cache
def get_client() -> EIARestAPIClient:
    api_key = os.environ["EIA_API_KEY"]
    return EIARestAPIClient(access_token=api_key)


def fetch_all_paginated_data(
    context: list[list[str]], parameters: dict
) -> pl.DataFrame:
    client = get_client()
    offset = 0
    rows = 5000
    total_results = []
    expected_rows = None

    while expected_rows is None or len(total_results) < expected_rows:
        response = client.execute_api_request(
            context, parameters={**parameters, "offset": offset, "length": rows}
        )

        expected_rows = int(response["response"]["total"])

        returned_records = response["response"]["data"]
        total_results.extend(returned_records)

        offset += rows

    return pl.DataFrame(total_results)


def fetch_net_generation_by_source(region: str = "US") -> pl.DataFrame:
    frame = fetch_all_paginated_data(
        ["electricity", "electric-power-operational-data", "data"],
        {
            "frequency": "monthly",
            "data": ["generation"],
            "facets": {"location": [region], "sectorid": ["99"]},
            "sort": [
                {"column": "period", "direction": "desc"},
                {"column": "fueltypeid", "direction": "desc"},
            ],
        },
    )

    casted_frame = (
        frame.lazy()
        .with_columns(
            pl.col("generation").cast(pl.Float32),
            (pl.col("fuelTypeDescription") + " (" + pl.col("fueltypeid") + ")").alias(
                "fuel_name"
            ),
        )
        .with_columns(pl.col("period").str.strptime(pl.Date, "%Y-%m"))
        .filter(pl.col("fueltypeid") != "ALL")
        .collect()
    )

    return casted_frame.pivot(on="fuel_name", index="period", values="generation")


def gen_ttm_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(pl.col("period")).with_columns(
        [pl.col(c).rolling_mean(window_size=12) for c in df.columns if c != "period"]
    )


@cache
def fetch_location_facet_values() -> list[LocationFacet]:
    client = get_client()
    response = client.execute_api_request(
        ["electricity", "electric-power-operational-data", "facet", "location"]
    )
    return [LocationFacet(**facet) for facet in response["response"]["facets"]]


def prep_scatterplot(in_x, indata, title, visible: bool):
    return go.Scatter(
        x=in_x,
        y=indata,
        mode="lines+markers",
        name=title,
        visible=visible,
        hoverlabel=dict(namelength=-1),
    )


def _scatterplots_for_frame(df: pl.DataFrame, visible: bool = True) -> list:
    return [
        go.Scatter(
            x=df["period"],
            y=df[c],
            mode="lines+markers",
            name=c,
            visible=visible,
            hoverlabel=dict(namelength=-1),
        )
        for c in df.columns
        if c != "period"
    ]


def _prompt_user_for_geography() -> LocationFacet:
    locations = fetch_location_facet_values()
    location_map = {l.id: l for l in locations}
    click.secho("Please select a two-letter location from the list below", fg="green")

    for l in locations:
        click.secho(f"- {l.alias}", fg="yellow")

    result = click.prompt("", prompt_suffix="> ")

    if result not in location_map:
        click.secho(f"{result} is not a known location. Try again.", fg="red")
        return _prompt_user_for_geography()

    return location_map[result]


@click.command()
@click.option(
    "--geography", "-g", help="If you already know the geography, enter it here"
)
@click.option("--outpath", "-o", help="Path to folder location of output html file")
def main(geography, outpath):
    load_dotenv()

    if geography is None:
        geography = _prompt_user_for_geography().id

    click.secho("Retrieving data")

    net_gen_by_source = fetch_net_generation_by_source(region=geography)
    net_gen_by_source_ttm = gen_ttm_dataframe(net_gen_by_source)

    click.secho("Building plots")

    non_ttm_scatterplots = _scatterplots_for_frame(net_gen_by_source, visible=False)
    ttm_scatterplots = _scatterplots_for_frame(net_gen_by_source_ttm)
    chart_data = non_ttm_scatterplots + ttm_scatterplots

    ttm_visible = ([False] * len(non_ttm_scatterplots)) + (
        [True] * len(ttm_scatterplots)
    )
    actual_visible = [not x for x in ttm_visible]

    updatemenus = list(
        [
            dict(
                active=0,
                buttons=list(
                    [
                        dict(
                            label="TTM",
                            method="update",
                            args=[{"visible": ttm_visible}],
                        ),
                        dict(
                            label="Actual",
                            method="update",
                            args=[{"visible": actual_visible}],
                        ),
                    ]
                ),
            )
        ]
    )

    footnote_text = "<i> Updated: " + datetime.today().strftime("%Y-%m-%d") + "</i>"
    layout = dict(
        title="Net Generation By Source - " + geography,
        xaxis=dict(title="Month"),
        yaxis=dict(title="thousand megawatthours"),
        hovermode="closest",
        updatemenus=updatemenus,
        annotations=[
            go.layout.Annotation(
                showarrow=False,
                text=footnote_text,
                xanchor="left",
                xref="paper",
                xshift=-5,
                x=0,
                yanchor="top",
                yref="paper",
                yshift=-15,
                y=0,
                font=dict(color="grey"),
            )
        ],
    )
    if outpath is not None:
        outfile = os.path.join(outpath, "Net Gen by Source - " + geography + ".html")
    else:
        outfile = "Net Gen by Source - " + geography + ".html"
    plotly.offline.plot(
        {"data": chart_data, "layout": layout}, filename=outfile, auto_open=False
    )

    click.secho("Complete")


if __name__ == "__main__":
    main()
