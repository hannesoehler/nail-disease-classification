# %pip install -Uqq fastai duckduckgo_search
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.text.all import *
from fastai.vision.all import *


def scrape_images(
    path,
    searches,
    max_n=30,
):
    """
    Scrape images from Google and save them in folder(s) named after the search
    term(s) provided as input list(s). If multiple search terms are provided per
    list, the first search term will be used as the representative name and the
    images for the other terms (synonyms) will be stored in subfolders of the
    representative name.

    Args:
        path (str, optional): Path to save images to.
        searches (list): List of lists containing search terms.
        max_n (int, optional): Max number of images to scrape per search term.
        Defaults to 30.
    """

    path = Path(path)
    for i, d_class in enumerate(searches):
        # first element of d_class is representative name of the disease class
        d_class_dir_name = d_class[0]
        d_dest = path / d_class_dir_name
        d_dest.mkdir(exist_ok=True, parents=True)
        for synonym in d_class:
            s_dest = path / d_class_dir_name / synonym
            s_dest.mkdir(exist_ok=True, parents=True)
            # if i == 3:  # for melanonychie, more images (50) were scraped
            #     urls = L(ddg_images(f"{synonym}", max_results=50)).itemgot("image")
            # else:
            #    urls = L(ddg_images(f"{synonym}", max_results=max_n)).itemgot("image")
            urls = L(ddg_images(f"{synonym}", max_results=max_n)).itemgot("image")
            download_images(s_dest, urls=urls)
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)


def rename_scraped_images(path):
    """Rename images according to the class they belong to (1st part of new
    name), the specific search term they were found under (2nd part), and an
    ascending number under specific search term (3rd part). Images that cannot
    be opened are deleted

    Args:
        path (str): path in which to look for disease classes folders
        and subdirectories with synonyms and images therein.
    """

    d_class_dir_name = [d for d in os.listdir(path) if not d.startswith(".")]

    for d_dir_name in d_class_dir_name:
        d_dir_path = path + "/" + d_dir_name
        synonyms = [d for d in os.listdir(d_dir_path) if not d.startswith(".")]
        for s_dir_name in synonyms:
            s_dir_path = d_dir_path + "/" + s_dir_name
            for index, imag in enumerate(os.listdir(s_dir_path)):
                # try to open image and rename if valid
                try:
                    img = Image.open(s_dir_path + "/" + imag)
                    img.verify()
                    os.rename(
                        s_dir_path + "/" + imag,
                        s_dir_path
                        + "/"
                        + d_dir_name
                        + "_"
                        + s_dir_name
                        + "_"
                        + str(index)
                        + ".jpg",
                    )
                # if not, remove image
                except:
                    print(
                        "Removed image that could not be opened:",
                        s_dir_path + "/" + imag,
                    )
                    os.remove(s_dir_path + "/" + imag)
                    continue


def copy_all_imgs_to_one_folder(
    path_old,
    path_new,
):
    """Copy all images from old folder (including subfolders) to new folder in
    order to have all images in one place without subfolders.

    Args:
        old_folder (str). Path to folder with subfolders.
        new_folder (str). Path to folder where all images will be copied to.
    """

    d_class_dir_name = [d for d in os.listdir(path_old) if not d.startswith(".")]

    for d_dir_name in d_class_dir_name:
        d_dir_path = path_old + "/" + d_dir_name
        synonyms = [d for d in os.listdir(d_dir_path) if not d.startswith(".")]
        for s_dir_name in synonyms:
            s_dir_path = d_dir_path + "/" + s_dir_name
            for imag in os.listdir(s_dir_path):
                shutil.copy(s_dir_path + "/" + imag, path_new)
