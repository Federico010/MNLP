"""
Module to load and prepare the dataset.

Useful functions:
- prepare_dataset
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests

from modules import paths, utils


def _get_sitelinks(entity_ids: Sequence[str], batch_size: int = 50) -> dict[str, Any]:
    """
    Use the Wikidata API to fetch sitelinks for a set of entity IDs.

    Args:
        entity_ids: sequence of entity IDs to fetch sitelinks for.
        batch_size: number of IDs to fetch in each API call.
    """

    sitelinks: dict[str, Any] = {}

    # Fetch sitelinks in batches
    for i in tqdm(range(0, len(entity_ids), batch_size), desc="Fetching batches"):
        batch_ids: Sequence[str] = entity_ids[i:i + batch_size]

        # API request
        url: str = 'https://www.wikidata.org/w/api.php'
        params: dict[str, str] = {
            'action': 'wbgetentities',
            'ids': '|'.join(batch_ids),
            'format': 'json',
            'props': 'sitelinks',
        }
        response: requests.Response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse the response
        entities: dict[str, Any] = response.json().get('entities', {})
        for entity_id, entity_data in entities.items():
            sitelinks[entity_id] = entity_data.get('sitelinks', {})

    return sitelinks


class _CommonPages:
    """
    Class to handle the common pages.

    Useful methods:
    - set
    - get
    """

    top_pages: tuple[str, ...] = tuple()

    @classmethod
    def set(cls, sitelinks: dict[str, dict[str, dict[str, Any]]], max_pages: int) -> None:
        """
        Class ,ethod to set the most common pages in the sitelinks.

        Args:
            sitelinks: dictionary of sitelinks.
            max_pages: maximim number of pages to consider. If <= 0, all pages will be considered.
        """

        # find the most common pages in the sitelinks
        page_counts: Counter = Counter(page for sitelink in sitelinks.values() for page in sitelink)

        # take the top pages
        top_pages_tuples: list[tuple[str, int]] = page_counts.most_common(max_pages if max_pages > 0 else None)
        cls.top_pages = tuple(page for page, _ in top_pages_tuples)
        print(f"Top {max_pages} pages: {cls.top_pages}")
    

    @classmethod
    def get(cls) -> tuple[str, ...]:
        """
        Class method to set the most common pages sorted by their number of sitelinks.
        """

        if not cls.top_pages:
            raise ValueError("Top pages not set. Call set() method first.")

        return cls.top_pages


def _group_by_page(sitelinks: dict[str, dict[str, dict[str, Any]]], pages: Iterable[str]) -> dict[str, dict[str, str]]:
    """
    Group the sitelinks by page.
    
    Args:
        sitelinks: dictionary of sitelinks.
        pages: pages to group by.

    Returns:
        Dictionary mapping each page to a dictionary of titles and their corresponding IDs.
    """

    page_to_title_to_id: dict[str, dict[str, str]] = {page: {} for page in pages}
    for id, sitelink in sitelinks.items():
        for page in pages:
            if page in sitelink:
                title: str = sitelink[page]['title']
                page_to_title_to_id[page][title] = id

    return page_to_title_to_id


def _get_common_page_lenght(sitelinks: dict[str, dict[str, dict[str, Any]]],
                           find_common_pages: bool = False,
                           max_pages: int = 0,
                           batch_size: int = 50
                           ) -> dict[str, dict[str, int]]:
    """
    Get for each id the number of characters in the most common pages.

    Args:
        sitelinks: dictionary of sitelinks.
        find_common_pages: whether to find the most common pages. Set to True to find the most common pages, otherwise the previously set pages will be used.
        max_pages: maximim number of pages to consider. If <= 0, all pages will be considered.
        batch_size: number of IDs to fetch in each API call.
    """

    if find_common_pages:
        # find the most common pages in the sitelinks
        _CommonPages.set(sitelinks, max_pages)
    top_pages: tuple[str, ...] = _CommonPages.get()

    # group the sitelinks by pages
    page_to_title_to_id: dict[str, dict[str, str]] = _group_by_page(sitelinks, top_pages)

    # map the pages names to their URLs
    site_map: dict[str, str] = utils.PageHandler.get_site_to_url()

    # make the requests in batches
    results: dict[str, dict[str, int]] = {}
    for page, titles_to_ids in page_to_title_to_id.items():
        titles: tuple[str, ...] = tuple(titles_to_ids.keys())
        for i in tqdm(range(0, len(titles_to_ids), batch_size), desc=f"Fetching {page} pages"):
            titles_batch: tuple[str, ...] = titles[i:i + batch_size]

            # API request
            url: str = f'{site_map[page]}/w/api.php'
            params: dict[str, str|bool] = {
                'action': 'query',
                'prop': 'info',
                'titles': '|'.join(titles_batch),
                'format': 'json'
            }
            response: requests.Response = requests.get(url, params=params)
            response.raise_for_status()

            # Parse the response
            parsed: dict[str, Any] = response.json()
            pages: dict[str, dict[str, Any]] = parsed.get('query', {}).get('pages', {})
            
            for data in pages.values():
                id: str = titles_to_ids[data['title']]
                if id not in results:
                    results[id] = {}
                results[id][page] = data.get('length', 0)

    return results


def prepare_dataset(split: Literal['train', 'valid']) -> pd.DataFrame:
    """
    Function to load and prepare the dataset.
    """

    output_file: Path = paths.UPDATED_TRAIN_SET if split == 'train' else paths.UPDATED_VALID_SET

    # Check if the updated dataset already exists
    if output_file.is_file():
        return pd.read_csv(output_file)

    # Load the dataset
    df: pd.DataFrame = pd.read_csv(f'hf://datasets/sapienzanlp/nlp2025_hw1_cultural_dataset/{split}.csv')
    df = df.head(50)  # For testing purposes, remove this line in production

    # Extract the IDs from the URLs and add them to the DataFrame
    df['id'] = df['item'].map(utils.extract_id)

    # take ids as a list
    ids: list[str] = df['id'].tolist()

    sitelinks: dict[str, Any] = _get_sitelinks(ids)

    # add the number of sitelinks to the DataFrame
    df['num_sitelinks'] = df['id'].map(lambda x: len(sitelinks.get(x, {})))

    # add the sitelinks lengths to the DataFrame
    find_common_pages: bool = split == 'train' # find common pages only during training
    common_page_lenght: dict[str, dict[str, int]] = _get_common_page_lenght(sitelinks, find_common_pages = find_common_pages, max_pages = 20)
    common_page_lenght_df: pd.DataFrame = pd.DataFrame.from_dict(common_page_lenght, orient='index').fillna(0).astype(int)
    df = df.merge(common_page_lenght_df, left_on='id', right_index=True, how='left')
    
    # Save the updated dataset to a new file
    df.to_csv(output_file, index=False)
    print(f"Updated dataset saved to {output_file}")
    return df
