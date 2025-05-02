"""
Module to load and prepare the dataset.

Useful functions:
- prepare_dataset

Imports: utils.dataset
"""

import asyncio
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import aiohttp
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm

from modules.utils import dataset as dataset_utils


async def _get_sitelinks(entity_ids: Sequence[str], batch_size: int = 50) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Use the Wikidata API to fetch sitelinks for a set of entity IDs asynchronously.

    Args:
        entity_ids: sequence of entity IDs to fetch sitelinks for.
        batch_size: number of IDs to fetch in each API call.
    """

    sitelinks: dict[str, dict[str, dict[str, Any]]] = {}

    async def fetch_batch(batch_ids: Sequence[str],
                          session: aiohttp.ClientSession
                          ) -> None:
        """
        Fetch a batch of sitelinks asynchronously and update the sitelinks dictionary.
        """

        async def make_request() -> None:
            """
            Make the API request and parse the response.
            """

            # API request to get the sitelinks
            url: str = 'https://www.wikidata.org/w/api.php'
            params: dict[str, str] = {
                'action': 'wbgetentities',
                'props': 'sitelinks',
                'ids': '|'.join(batch_ids),
                'format': 'json'
            }

            # Parse the response
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                entities: dict[str, dict[str, Any]] = (await response.json()).get('entities', {})
                for entity_id, entity_data in entities.items():
                    sitelinks[entity_id] = entity_data.get('sitelinks', {})
        
        await make_request()

    # Fetch the sitelinks in batches
    async with aiohttp.ClientSession() as session:
        await async_tqdm.gather(
            *(fetch_batch(entity_ids[i:i + batch_size], session) for i in range(0, len(entity_ids), batch_size)),
            desc = "Fetching sitelinks",
            unit = "requests"
        )

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

        # Find the most common pages in the sitelinks
        page_counts: Counter = Counter(page for sitelink in sitelinks.values() for page in sitelink)

        # Take the top pages
        top_pages_tuples: list[tuple[str, int]] = page_counts.most_common(max_pages if max_pages > 0 else None)
        cls.top_pages = tuple(page for page, _ in top_pages_tuples)
        print(f"Top {len(cls.top_pages)} pages: {cls.top_pages}")
    

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


async def _get_common_pages_features(sitelinks: dict[str, dict[str, dict[str, Any]]],
                              batch_size: int = 50,
                              concurrent_requests: int = 10
                              ) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Get for each id the features in the most common pages asynchronously.

    The features are the number of characters, the number of links and the number of external links.
    The features are stored in a dictionary with the ID as the key and a dictionary of features as the value.

    Args:
        sitelinks: dictionary of sitelinks.
        batch_size: number of IDs to fetch in each API call.
        concurrent_requests: number of concurrent requests to make.
    """

    # Group the sitelinks by pages, considering only the common pages
    page_to_title_to_id: dict[str, dict[str, str]] = _group_by_page(sitelinks, _CommonPages.get())

    # Map the pages names to their URLs
    site_map: dict[str, str] = await dataset_utils.PageHandler.get_site_to_url()

    # Queue to hold the requests
    queue = asyncio.Queue()

    # Dictionary to hold the results
    results: dict[str, dict[str, dict[str, Any]]] = {}
    results_lock: asyncio.Lock = asyncio.Lock()

    # Progress bar
    print(f"The progress bar will slow down with time, as the number of requests increases.")
    pbar: async_tqdm = async_tqdm(desc = "Fetching infos in different languages", bar_format = "{l_bar}{bar}| [{elapsed}, {rate_fmt}]", unit = "requests")
    pbar_lock: asyncio.Lock = asyncio.Lock()

    async def worker(session: aiohttp.ClientSession) -> None:
        """
        Worker function to process requests from the queue.
        """

        while not queue.empty():
            
            # Extract the task parameters
            page: str
            titles_batch: tuple[str, ...]
            titles_to_ids: dict[str, str]
            continue_params: dict[str, str]
            page, titles_batch, titles_to_ids, continue_params = await queue.get()

            url: str = f'{site_map[page]}/w/api.php'
            params: dict[str, str | bool] = {
                'action': 'query',
                'prop': 'info|images|templates|categories|iwlinks|langlinks|redirects|extracts',
                'titles': '|'.join(titles_batch),
                'imlimit': 'max',
                'tllimit': 'max',
                'cllimit': 'max',
                'iwlimit': 'max',
                'lllimit': 'max',
                'rdlimit': 'max',
                'exlimit': 'max',
                'exintro': '',
                'explaintext': '',
                'format': 'json'
            }
            params.update(continue_params)

            # Parse the response
            async with session.get(url, params = params) as response:
                response.raise_for_status()
                parsed: dict[str, Any] = await response.json()
                pages: dict[str, dict[str, Any]] = parsed.get('query', {}).get('pages', {})

                for data in pages.values():
                    id: str = titles_to_ids[data['title']]
                    async with results_lock:
                        if id not in results:
                            results[id] = {}
                        if page not in results[id]:
                            results[id][page] = {}
                        
                        # Update the results with the new data
                        old_length: int = results[id][page].get('length', 0)
                        results[id][page]['length'] = old_length + data.get('length', 0)

                        old_images_count: int = results[id][page].get('images_count', 0)
                        results[id][page]['images_count'] = old_images_count + len(data.get('images', []))

                        old_templates_count: int = results[id][page].get('templates_count', 0)
                        results[id][page]['templates_count'] = old_templates_count + len(data.get('templates', []))

                        old_categories_count: int = results[id][page].get('categories_count', 0)
                        results[id][page]['categories_count'] = old_categories_count + len(data.get('categories', []))

                        old_iwlinks_count: int = results[id][page].get('iwlinks_count', 0)
                        results[id][page]['iwlinks_count'] = old_iwlinks_count + len(data.get('iwlinks', []))

                        old_langlinks_count: int = results[id][page].get('langlinks_count', 0)
                        results[id][page]['langlinks_count'] = old_langlinks_count + len(data.get('langlinks', []))

                        old_redirects_count: int = results[id][page].get('redirects_count', 0)
                        results[id][page]['redirects_count'] = old_redirects_count + len(data.get('redirects', []))

                        old_extract: str = results[id][page].get('extract', '')
                        results[id][page]['extract'] = old_extract + data.get('extract', '')

                # Add the next continue request to the queue
                if 'continue' in parsed:
                    async with pbar_lock:
                        pbar.total += 1
                    await queue.put((page, titles_batch, titles_to_ids, parsed['continue']))

            # Update the progress bar and mark the task as done
            pbar.update(1)
            queue.task_done()

    # Process the requests
    async with aiohttp.ClientSession() as session:

        # Starting requests
        for page, titles_to_ids in page_to_title_to_id.items():
            titles: tuple[str, ...] = tuple(titles_to_ids.keys())
            for i in range(0, len(titles), batch_size):
                await queue.put((page, titles[i:i + batch_size], titles_to_ids, {}))
        pbar.total = queue.qsize()
        pbar.refresh()

        # Wait for tasks to be done
        await asyncio.gather(*(asyncio.create_task(worker(session)) for _ in range(concurrent_requests)),
                             return_exceptions = True
                             )

    return results


def extract_dataset(split: Literal['train', 'validation', 'test']) -> pd.DataFrame:
    """
    Function to load the dataset and add the features.
    """

    # Files paths
    original_file: Path|str
    output_file: Path
    original_file, output_file = dataset_utils.get_split_paths(split)

    # Check if the updated dataset already exists
    if output_file.is_file():
        return pd.read_csv(output_file, index_col = 'id', keep_default_na = False)

    # Load the dataset
    df: pd.DataFrame = pd.read_csv(original_file)

    # Extract the IDs from the URLs and set them as the index
    ids: pd.Series[str] = df['item'].map(dataset_utils.extract_id)
    df = df.set_index(ids)
   
    # Get the sitelinks for each id
    sitelinks: dict[str, dict[str, dict[str, Any]]] = asyncio.run(_get_sitelinks(ids.tolist()))

    # Set the most common pages
    if split == 'train':
        _CommonPages.set(sitelinks, max_pages = 10)

    # Create a dataframe with the new features
    common_pages_features: dict[str, dict[str, dict[str, Any]]] = asyncio.run(_get_common_pages_features(sitelinks))
    flattened_dict: dict[str, dict[str, Any]] = dataset_utils.flatten_dict(common_pages_features)
    common_pages_features_df: pd.DataFrame = pd.DataFrame.from_dict(flattened_dict, orient = 'index')

    # Fill NaN and convert when necessary
    numerical_columns: pd.Index[str] = common_pages_features_df.select_dtypes(include = 'number').columns
    string_columns: pd.Index[str] = common_pages_features_df.select_dtypes(include = 'object').columns
    common_pages_features_df[numerical_columns] = common_pages_features_df[numerical_columns].fillna(0).astype(int)
    common_pages_features_df[string_columns] = common_pages_features_df[string_columns].fillna('')

    # Add the new features to the original DataFrame
    df['sitelinks_count'] = df.index.map(lambda id: len(sitelinks.get(id, {})))
    df = pd.concat([df, common_pages_features_df], axis = 1)
    
    # Save the updated dataset
    df.to_csv(output_file, index_label = 'id')
    print(f"Dataset saved to {output_file}")

    return df
