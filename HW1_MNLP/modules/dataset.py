"""
Module to load and prepare the dataset.

USEFUL FUNCTIONS:
- prepare_dataset
"""

from typing import Any, Literal

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests

from modules import paths, utils


def get_sitelinks(entity_ids: list[str], batch_size: int = 50) -> dict[str, Any]:
    """
    Use the Wikidata API to fetch sitelinks for a list of entity IDs.
    """

    sitelinks: dict[str, Any] = {}

    # Fetch sitelinks in batches
    for i in tqdm(range(0, len(entity_ids), batch_size), desc="Fetching batches"):
        batch_ids: list[str] = entity_ids[i:i + batch_size]

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


def get_common_pages(sitelinks: dict[str, dict[str, dict[str, Any]]], num_pages: int) -> list[str]:
    """
    Get the most common pages in the sitelinks.
    """

    # find the most common pages in the sitelinks
    page_counts: dict[str, int] = {}
    for sitelink in sitelinks.values():
        for page in sitelink:
            page_counts[page] = page_counts.get(page, 0) + 1

    # take the top pages
    sorted_pages: list[tuple[str, int]] = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)
    top_pages: list[str] = [lang for lang, _ in sorted_pages[:num_pages]]
    print(f"Top {num_pages} pages: {top_pages}")

    return top_pages


def get_common_page_lenght(sitelinks: dict[str, dict[str, dict[str, Any]]], num_pages: int = 20, batch_size: int = 50) -> dict[str, dict[str, int]]:
    """
    Get for each id the number of characters in the most common pages.
    """

    top_pages: list[str] = get_common_pages(sitelinks, num_pages)

    # group by pages in order to make batch requests for each different page
    page_to_title_to_id: dict[str, dict[str, str]] = {page: {} for page in top_pages}
    for id, sitelink in sitelinks.items():
        for page in top_pages:
            if page in sitelink:
                title: str = sitelink[page]['title']
                page_to_title_to_id[page][title] = id

    # map the pages names to their URLs
    site_map: dict[str, str] = utils.site_to_url()

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

    # Extract the IDs from the URLs and add them to the DataFrame
    df['id'] = df['item'].map(utils.extract_id)

    # take ids as a list
    ids: list[str] = df['id'].tolist()

    sitelinks: dict[str, Any] = get_sitelinks(ids)

    # add the number of sitelinks to the DataFrame
    df['num_sitelinks'] = df['id'].map(lambda x: len(sitelinks.get(x, {})))

    # add the sitelinks lengths to the DataFrame
    common_page_lenght: dict[str, dict[str, int]] = get_common_page_lenght(sitelinks)
    common_page_lenght_df: pd.DataFrame = pd.DataFrame.from_dict(common_page_lenght, orient='index').fillna(0).astype(int)
    df = df.merge(common_page_lenght_df, left_on='id', right_index=True, how='left')
    
    # Save the updated dataset to a new file
    df.to_csv(output_file, index=False)
    print(f"Updated dataset saved to {output_file}")
    return df
