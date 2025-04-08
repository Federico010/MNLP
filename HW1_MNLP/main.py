import pandas as pd
from tqdm import tqdm
from wikidata.client import Client

from modules import paths


def extract_id(url: str) -> str:
    """
    Extract the id from the wikidata url
    """

    return url.split('/')[-1]

def get_num_languages(id: str, client: Client) -> int|None:
    """
    Find the number of languages in wich wikipedia has a page for the entity
    """

    try:
        entity = client.get(id)
        sitelinks = entity.attributes.get('sitelinks', {})
        num_languages: int = 0
        for _, data in sitelinks.items():
            if '.wikipedia.org' in data['url']:
                num_languages += 1
        return num_languages
    except Exception as e:
        print(f"Error processing entity {id}: {e}")
        return None


def main() -> None:
    # load the dataset
    df: pd.DataFrame = pd.read_csv(paths.TRAINING_SET, sep='\t')

    # exctract the id from the url
    df['id'] = df['item'].apply(extract_id)


    # add the column with the number of languages
    client: Client = Client()
    tqdm.pandas(desc="Processing entities")
    df['num_languages'] = df['id'].progress_apply(lambda id: get_num_languages(id, client))

    # save the updated dataset to a new file
    output_file = "updated_dataset.csv"
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Updated dataset saved to {output_file}")


if __name__ == '__main__':
    main()
