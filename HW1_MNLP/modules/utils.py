"""
Module containing utility functions for the whole project.

Useful functions:
- extract_id

Useful classes:
- PageHandler
"""

import aiohttp
import asyncio
from typing import Any


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
    _lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    async def _get_sitematrix() -> dict[str, Any]:
        """
        Async static method to get the sitematrix.
        """

        # API request to get the sitematrix
        url: str = 'https://meta.wikimedia.org/w/api.php'
        params: dict[str, str] = {
            'action': 'sitematrix',
            'format': 'json'
        }

        # Parse the response
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data: dict[str, Any] = await response.json()

        return data.get('sitematrix', {})


    @classmethod
    async def get_site_to_url(cls) -> dict[str, str]:
        """
        Async class method that returns a dictionary mapping site names (dbname) to their URLs.
        """

        # Return the cached value if it exists
        if cls.site_to_url:
            return cls.site_to_url

        # Create the dictionary starting from the sitematrix
        async with cls._lock:
            if not cls.site_to_url:  # Double-check to avoid race conditions
                sitematrix: dict[str, Any] = await cls._get_sitematrix()
                for key, val in sitematrix.items():
                    if key.isdigit():
                        for site in val.get('site', []):
                            cls.site_to_url[site['dbname']] = site['url']
                    elif key == 'specials':
                        for site in val:
                            cls.site_to_url[site['dbname']] = site['url']

        return cls.site_to_url
