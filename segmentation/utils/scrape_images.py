# %pip install -Uqq fastai duckduckgo_search
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.text.all import *
from fastai.vision.all import *


def scrape_images(
    dir_name,
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
        dir_name (str, optional): Path to save images to.
        searches (list): List of lists containing search terms.
        max_n (int, optional): Max number of images to scrape per search term.
        Defaults to 30. Note, for melanonychie, 50 images are scraped bc the
        initial number was too low.
    """

    # TODO: Maybe use cleaner code with pathlib: Path(dir.name).parents[2]
    path = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dir_name
        )
    )

    # Seach for Urls and download images
    for i, d_class in enumerate(searches):
        # first element of d_class is going to be the representative name of the
        # class
        d_class_dir_name = d_class[0]
        d_dest = path / d_class_dir_name
        d_dest.mkdir(exist_ok=True, parents=True)
        for synonym in d_class:
            s_dest = path / d_class_dir_name / synonym
            s_dest.mkdir(exist_ok=True, parents=True)
            if i == 3:  # more images for melanonychie
                urls = L(ddg_images(f"{synonym}", max_results=50)).itemgot("image")
            else:
                urls = L(ddg_images(f"{synonym}", max_results=max_n)).itemgot("image")
            download_images(s_dest, urls=urls)

    # Removing images that might not have been downloaded properly
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)


def rename_scraped_images(
    dir_name="data/testset/imgs_scraped",
):
    """Rename images according to the class they belong to (1st part of new
    name), the specific search term they were found under (2nd part), and an
    ascending number under specific search term (3rd part). Images that cannot
    be opened are deleted and not renamed.

    Args:
        dir_name (str, optional): folder in which to look for classes folders
        and subdirectories with synonyms and images therein. Defaults to
        "data/imgs_scraped".
    """

    # path where images are located
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dir_name
    )

    # get directories in path without hidden directories
    d_class_dir_name = [d for d in os.listdir(path) if not d.startswith(".")]

    for d_dir_name in d_class_dir_name:
        d_dir_path = path + "/" + d_dir_name
        synonyms = [d for d in os.listdir(d_dir_path) if not d.startswith(".")]
        for s_dir_name in synonyms:
            s_dir_path = d_dir_path + "/" + s_dir_name
            for index, imag in enumerate(os.listdir(s_dir_path)):
                # try to open image; if valid image rename
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
    old_folder="data/testset/imgs_scraped", new_folder="data/testset/imgs_scraped_clean"
):
    """Copy all images from old folder (including subfolders) to new folder, to
    have all images in one place.

    Args:
        old_folder (str, optional). Defaults to "data/imgs_scraped".
        new_folder (str, optional). Defaults to "data/imgs_scraped_clean".
    """

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), old_folder
    )
    path_new = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), new_folder
    )

    d_class_dir_name = [d for d in os.listdir(path) if not d.startswith(".")]

    for d_dir_name in d_class_dir_name:
        d_dir_path = path + "/" + d_dir_name  #
        synonyms = [d for d in os.listdir(d_dir_path) if not d.startswith(".")]
        for s_dir_name in synonyms:
            s_dir_path = d_dir_path + "/" + s_dir_name
            for imag in os.listdir(s_dir_path):
                shutil.copy(s_dir_path + "/" + imag, path_new)
