import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def sampling_percent(df, percent):
    df = df.sample(frac=1, random_state=np.random.seed())

    return df.sample(frac=percent)


def sampling_fixed_number(df, fixed_number):
    df = df.sample(frac=1, random_state=np.random.seed())

    if fixed_number > len(df):
        return df.sample(n=len(df))
    return df.sample(n=fixed_number)


def load_uvp5(path):

    all_tsv_files = find_files(path, '.csv')
    print(path)
    # TODO
    metadata = pd.read_csv(all_tsv_files[0], sep='\t')

    files = find_files(path, '.bmp')
    file_names = [os.path.basename(path) for path in files]

    df = pd.DataFrame(file_names, columns=['File Name'])
    df['path'] = [os.path.join(path, file) for file in files]

    df['uvp_model'] = ['UVP5'] * len(df)
    df['depth'] = ['NaN'] * len(df)
    df['lat'] = ['NaN'] * len(df)
    df['lon'] = ['NaN'] * len(df)

    return df


def copy_image_from_df(df, out_dir, target_size=None, cutting_ruler=False, invert_img=True):

    if target_size is None:
        target_size = [227, 227]

    rows_to_drop = []
    for index, row in tqdm(df.iterrows()):
        image_path = row['path']
        if not os.path.exists(image_path):
            rows_to_drop.append(index)  # Store index of row to drop
            continue
        image_filename = os.path.basename(image_path)
        image_filename = image_filename.replace('output/', '', 1)
        target_path = os.path.join(out_dir, image_filename)
        dir_path = os.path.dirname(target_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        img = Image.open(image_path)

        if cutting_ruler:
            # crop 31px from bottom
            width, height = img.size
            right = width
            bottom = height - 31
            cropped_img = img.crop((0, 0, right, bottom))
        else:
            cropped_img = img

        if invert_img:
            # invert image
            img_gray = cropped_img.convert("L")
            img_array = np.array(img_gray)
            max_value = np.iinfo(img_array.dtype).max
            inverted_array = max_value - img_array
            inverted_img = Image.fromarray(inverted_array)
        else:
            inverted_img = cropped_img
        # resize image
        resized_image = inverted_img.resize((target_size[0], target_size[1]), resample=Image.Resampling.LANCZOS)
        resized_image.save(target_path)

    df.drop(rows_to_drop, inplace=True)
    return df

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