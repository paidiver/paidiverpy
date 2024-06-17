import os


def find_files(path, fmt='.tsv'):
    """
    Find all .tsv files in the specified directory.

    Args:
    path (str): The directory path where to look for .tsv files.

    Returns:
    list: A list of paths to .tsv files found within the directory.
    """
    all_files = []
    # Check if the given path is a valid directory
    if not os.path.isdir(path):
        print("Provided path is not a directory.")
        return []

    # Walk through all directories and files within the provided path
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(fmt):
                all_files.append(os.path.join(root, file))

    return all_files