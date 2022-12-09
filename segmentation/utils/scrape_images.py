# %pip install -Uqq fastai duckduckgo_search
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.text.all import *
from fastai.vision.all import *


def scrape_images(
    dir_name="data/testset/imgs_scraped",
    searches=[
        ["Nagel gesund", "Nagel normal", "Fingernagel", "Fußnagel"],
        ["Onychomykose Nagel", "Nagelmykose", "Nagelpilz"],
        ["Dystrophie Nagel", "Nageldystrophie", "Onychodystrophie"],
        [
            "Melanonychie Nagel",
            "Streifenförmige Nagelpigmentierung",
            "Longitudinale Melanonychie",
        ],
        ["Onycholyse Nagel", "Nagelablösung", "Nagelabhebung"],
    ],
    max_n=50,
):
    """
    Scrape images from Google and save them in folder(s) named after the search
    term(s) provided as input list(s). If multiple search terms are provided per
    list, the first search term will be used as the representative name of the
    class and the images will in subfolders of this class named after the
    specific search terms.

    Args:
        dir_name (str, optional): Path to save images to. Defaults to
        "data/imgs_scraped". searches (list, optional): List all labels to search for
        including synonyms. Defaults to ... max_n (int, optional): Maximum
        number of images to download per label. Defaults to 50.
    """

    # TODO: Maybe use cleaner code with pathlib: Path(dir.name).parents[2]
    path = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dir_name
        )
    )

    # Seach for Urls and download images
    for d_class in searches:
        # first element of d_class is going to be the representative name of the
        # class
        d_class_dir_name = d_class[0]
        d_dest = path / d_class_dir_name
        d_dest.mkdir(exist_ok=True, parents=True)
        for synonym in d_class:
            s_dest = path / d_class_dir_name / synonym
            s_dest.mkdir(exist_ok=True, parents=True)
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
    ascending number under specific search term (3rd part).

    Args:
        dir_name (str, optional): folder in which to look for classes folders
        and subdirectories with synonyms and images therein. Defaults to
        "data/imgs_scraped".
    """

    home_dir = os.getcwd()

    # path where images are located
    # path = home_dir + "/" + dir_name + "/"
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


def copy_all_imgs_to_one_folder(
    old_folder="data/testset/imgs_scraped", new_folder="data/imgs_scraped_clean"
):
    """Copy all images from old folder (including subfolders) to new folder, to
    have all images in one folder.

    Args:
        old_folder (str, optional). Defaults to "data/imgs_scraped".
        new_folder (str, optional). Defaults to "data/imgs_scraped_clean".
    """

    # home_dir = os.getcwd()
    # path = home_dir + "/" + old_folder + "/"
    # path_new = home_dir + "/" + new_folder + "/"

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
