"""
Module containing utility functions for the whole project.

USEFUL FUNCTIONS:
- extract_id
"""

def extract_id(url: str) -> str:
    """
    Extract the id from the Wikidata URL.
    """

    return url.split('/')[-1]
