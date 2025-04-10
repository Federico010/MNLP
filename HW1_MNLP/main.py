"""
Starting point for the script.
"""

from modules import dataset


def main() -> None:
    """
    Main function to run the script.
    """

    # Prepare the training set
    dataset.prepare_dataset('train')

    # Prepare the validation set
    dataset.prepare_dataset('valid')


if __name__ == '__main__':
    main()
