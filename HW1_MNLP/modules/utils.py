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


class PageHandler:
    """
    Class to handle the pages.
    """

    site_to_url: dict[str, str] = {}

    @staticmethod
    def _get_sitematrix() -> dict[str, Any]:
        """
        Get the sitematrix.
        """

        url: str = 'https://meta.wikimedia.org/w/api.php'
        params: dict[str, str] = {
            'action': 'sitematrix',
            'format': 'json'
        }
        response: requests.Response = requests.get(url, params=params)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # Parse the response to get the sitematrix
        return data.get('sitematrix', {})

    @classmethod
    def get_site_to_url(cls) -> dict[str, str]:
        """
        Return a dictionary mapping site names (dbname) to their URLs.
        """

        if cls.site_to_url:
            return cls.site_to_url

        sitematrix: dict[str, Any] = PageHandler._get_sitematrix()
        for key, val in sitematrix.items():
            if key.isdigit():
                for site in val.get('site', []):
                    cls.site_to_url[site['dbname']] = site['url']
            elif key == 'specials':
                for site in val:
                    cls.site_to_url[site['dbname']] = site['url']

        return cls.site_to_url
