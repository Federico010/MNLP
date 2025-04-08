import pandas as pd
from wikidata.client import Client
from tqdm import tqdm  # Importa tqdm per la barra di progresso
from modules import paths


def extract_entity_id(item_url: str) -> str:
    """
    Estrae l'ID dell'entità (es. Q306) dal campo item contenente l'URL.
    """
    return item_url.split('/')[-1]  # Prende l'ultima parte dell'URL


def get_num_languages(entity_id: str, client: Client) -> int|None:
    """
    Calcola il numero di lingue in cui un'entità è tradotta su Wikipedia.
    """
    try:
        entity = client.get(entity_id)  # Ottieni l'entità da Wikidata
        sitelinks = entity.attributes.get('sitelinks', {})  # Collegamenti ai siti Wikimedia
        """num_languages = 0
        for _, data in sitelinks.items():
            if '.wikipedia.org' in data['url']:
                num_languages += 1
        return num_languages  # Ritorna il numero di lingue"""
        return len(sitelinks)
    except Exception as e:
        print(f"Error processing entity {entity_id}: {e}")
        return None  # Ritorna 0 in caso di errore


def main() -> None:
    # Caricamento del dataset
    df: pd.DataFrame = pd.read_csv(paths.TRAINING_SET, sep='\t')
    print("Dataset loaded:")
    print(df.head())

    # Estrai l'ID dell'entità dal campo 'item'
    df['entity_id'] = df['item'].apply(extract_entity_id)

    # Inizializza il client di Wikidata
    client = Client()

    # Calcola il numero di lingue per ogni riga del dataset con tqdm
    tqdm.pandas(desc="Processing entities")  # Inizializza tqdm con una descrizione
    df['num_languages'] = df['entity_id'].progress_apply(lambda eid: get_num_languages(eid, client))

    # Salva il dataset aggiornato in un nuovo file
    output_file = "updated_dataset.csv"
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Updated dataset saved to {output_file}")


if __name__ == '__main__':
    main()