"""
Module containing utility functions for the whole project.

Useful functions:
- extract_id

Useful classes:
- PageHandler
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

    Useful methods:
    - get_site_to_url
    """

    site_to_url: dict[str, str] = {}

    @staticmethod
    def _get_sitematrix() -> dict[str, Any]:
        """
        Static method to get the sitematrix.
        """

        # API request to get the sitematrix
        url: str = 'https://meta.wikimedia.org/w/api.php'
        params: dict[str, str] = {
            'action': 'sitematrix',
            'format': 'json'
        }
        response: requests.Response = requests.get(url, params=params)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # Parse the response
        return data.get('sitematrix', {})


    @classmethod
    def get_site_to_url(cls) -> dict[str, str]:
        """
        Class method that returns a dictionary mapping site names (dbname) to their URLs.
        """

        # return the cached value if it exists
        if cls.site_to_url:
            return cls.site_to_url

        # create the dictionary starting from the sitematrix
        sitematrix: dict[str, Any] = PageHandler._get_sitematrix()
        for key, val in sitematrix.items():
            if key.isdigit():
                for site in val.get('site', []):
                    cls.site_to_url[site['dbname']] = site['url']
            elif key == 'specials':
                for site in val:
                    cls.site_to_url[site['dbname']] = site['url']

        return cls.site_to_url
