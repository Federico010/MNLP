"""
Module containing utility functions for the dataset creation.

Useful functions:
- get_split_paths
- extract_id
- flatten_dict

Useful classes:
- PageHandler

Imports: paths
"""

import asyncio
from pathlib import Path
from typing import Any, Literal

import aiohttp

from modules import paths


def get_split_paths(split: Literal['train', 'validation', 'test']) -> tuple[str|Path, Path]:
    """
    Function to get the paths (or URI) of the original and updated datasets.
    """

    if split == 'train':
        return paths.TRAIN_SET, paths.UPDATED_TRAIN_SET
    elif split == 'validation':
        return paths.VALIDATION_SET, paths.UPDATED_VALIDATION_SET
    else:
        return paths.TEST_SET, paths.UPDATED_TEST_SET


def extract_id(url: str) -> str:
    """
    Extract the id from the Wikidata URL.
    """

    return url.split('/')[-1]


def flatten_dict(d: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """
    Flatten a nested dictionary to make it suitable for pd.DataFrame.from_dict().
    """

    flattened_dict: dict[str, dict[str, int]] = {}
    for key, subdict in d.items():
        if key not in flattened_dict:
            flattened_dict[key] = {}
        for subkey, features in subdict.items():
            for feature, value in features.items():
                flattened_dict[key][f'{subkey}_{feature}'] = value
    return flattened_dict


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
            if not cls.site_to_url:  # Avoid race conditions
                sitematrix: dict[str, Any] = await cls._get_sitematrix()
                for key, val in sitematrix.items():
                    if key.isdigit():
                        for site in val.get('site', []):
                            cls.site_to_url[site['dbname']] = site['url']
                    elif key == 'specials':
                        for site in val:
                            cls.site_to_url[site['dbname']] = site['url']

        return cls.site_to_url
