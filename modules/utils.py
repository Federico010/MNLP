"""
Module containing utility functions for the whole project.

Useful functions:
- extract_id
- plot_confusion_matrix

Useful classes:
- PageHandler
"""

import asyncio
from typing import Any

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


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


def plot_confusion_matrix(true_y: ArrayLike, pred_y: ArrayLike, label_encoder: LabelEncoder|None = None) -> None:
    """
    Plot the confusion matrix.
    """

    # Get the confusion matrix
    confusion: NDArray[np.int_] = confusion_matrix(true_y, pred_y)

    # Plot the confusion matrix
    sns.heatmap(confusion, annot = True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Replace the labels
    if label_encoder is not None:
        plt.xticks(labels = label_encoder.classes_.tolist(),
                   ticks = np.arange(len(label_encoder.classes_)) + 0.5,
                   rotation = 45
                   )
        plt.yticks(labels = label_encoder.classes_.tolist(),
                   ticks = np.arange(len(label_encoder.classes_)) + 0.5,
                   rotation = 45
                   )

    plt.show()
