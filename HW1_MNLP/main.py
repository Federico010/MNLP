from typing import Any, Literal

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests

from modules import paths, utils


def get_data_batch(entity_ids: list[str], props: str) -> dict[str, Any]:
    """
    Use the Wikidata API to fetch data for a batch of entity IDs.
    """

    url: str = 'https://www.wikidata.org/w/api.php'
    params: dict[str, str] = {
        'action': 'wbgetentities',
        'ids': '|'.join(entity_ids),
        'format': 'json',
        'props': props
    }
    response: requests.Response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get('entities', {})


# FUNZIONE DA MODIFICARE PER ESTRARRE I NUOVI DATI
#
# SE SERVE PIÙ DI UNA RIGA PER PROPRIETÀ, CREARE UNA NUOVA FUNZIONE
# (es.: get_num_sitelinks anzichè len(entity_data.get('sitelinks', {}))
def process_batch(entity_ids: list[str]) -> list[dict[str, Any]]:
    """
    Process a batch of entity IDs and return a structured dictionary.
    """

    entities: dict[str, Any] = get_data_batch(entity_ids, props='sitelinks')
    results: list[dict[str, Any]] = []
    for entity_id, entity_data in entities.items():
        num_sitelinks: int = len(entity_data.get('sitelinks', {}))
        results.append({
            'id': entity_id,
            'num_sitelinks': num_sitelinks
            # add here any other properties you want to extract
        })
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
    df['id'] = df['item'].apply(utils.extract_id)

    # Process entities in batches
    batch_size: int = 50
    all_ids: list[str] = df['id'].tolist()
    new_data: list[dict[str, Any]] = []

    # show progress bar
    tqdm.pandas(desc="Processing entities")
    for i in tqdm(range(0, len(all_ids), batch_size), desc="Fetching batches"):
        # save the new data to a list
        batch_ids: list[str] = all_ids[i:i + batch_size]
        batch_results: list[dict[str, Any]] = process_batch(batch_ids)
        new_data.extend(batch_results)
    
    # Merge the new data with the original DataFrame
    df = df.merge(pd.DataFrame(new_data), on='id', how='left')

    # Save the updated dataset to a new file
    df.to_csv(output_file, index=False)
    print(f"Updated dataset saved to {output_file}")

    return df


def main() -> None:
    """
    Main function to run the script.
    """

    # Prepare the training set
    prepare_dataset('train')

    # Prepare the validation set
    prepare_dataset('valid')


if __name__ == '__main__':
    main()
