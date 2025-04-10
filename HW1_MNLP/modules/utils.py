"""
Module containing utility functions for the whole project.

USEFUL FUNCTIONS:
- extract_id
- site_to_url
"""

from typing import Any

import requests


def extract_id(url: str) -> str:
    """
    Extract the id from the Wikidata URL.
    """

    return url.split('/')[-1]


def site_to_url() -> dict[str, str]:
    """
    Return a dictionary mapping site names (dbname) to their URLs.
    """
    url: str = 'https://meta.wikimedia.org/w/api.php'
    params: dict[str, str] = {
        'action': 'sitematrix',
        'format': 'json'
    }
    response: requests.Response = requests.get(url, params=params)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    # Parse the response to get the site names and URLs
    site_map: dict[str, str] = {}
    sitematrix: dict[str, Any] = data.get('sitematrix', {})

    for key, val in sitematrix.items():
        if key.isdigit():
            for site in val.get('site', []):
                site_map[site['dbname']] = site['url']
        elif key == 'specials':
            for site in val:
                site_map[site['dbname']] = site['url']

    return site_map