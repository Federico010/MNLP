"""
Starting point for the script.

Imports: dataset, graph
"""

from pandas import DataFrame

from modules import dataset, graph


def main() -> None:
    """
    Main function to run the script.
    """

    # Prepare the training set
    train_set: DataFrame = dataset.prepare_dataset('train')

    # da scrivere meglio nel prepare dataset
    train_set = train_set.select_dtypes(include=['number'])
    train_set.drop(columns=['num_sitelinks'], inplace=True)

    for mode, treshold in (('iou', 0.5), ('correlation', 0.45), ('filtered correlation', 0.4)):
        # Calculate the similarity graph
        graph.get_similarity_graph(train_set, similarity_threshold=treshold, mode=mode, save_fig=True)

    # Prepare the validation set
    dataset.prepare_dataset('valid')


if __name__ == '__main__':
    main()
