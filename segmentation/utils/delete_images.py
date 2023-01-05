import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from numpy import dot
from numpy.linalg import norm
from PIL import Image
from collections import defaultdict


class Img2Vec:  # https://github.com/christiansafka/img2vec
    RESNET_OUTPUT_SIZES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    def __init__(
        self, cuda=False, model="resnet-34", layer="default", layer_output_size=512
    ):
        """Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name == "alexnet":
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name == "alexnet":
                    return my_embedding.numpy()[:, :]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = (
                self.normalize(self.to_tensor(self.scaler(img)))
                .unsqueeze(0)
                .to(self.device)
            )

            if self.model_name == "alexnet":
                my_embedding = torch.zeros(1, self.layer_output_size)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name == "alexnet":
                    return my_embedding.numpy()[0, :]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name.startswith("resnet") and not model_name.startswith("resnet-"):
            model = getattr(models, model_name)(pretrained=True)
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == "resnet-34":
            model = models.resnet34(pretrained=True)
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            if layer == "default":
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError("Model %s was not found" % model_name)


def cosine_similarity(a, b):

    """Cosine similarity between vectors a and b

    Args:
        a: vector a
        b: vector b

    Returns:
        cosine similarity between a and b
    """

    return dot(a, b) / (norm(a) * norm(b))


def delete_duplicate_images(path, similarity_thesh):

    """Delete duplicate images from a directory using cosine similarity

    Args:
        path: path to directory containing images
        similarity_thesh: threshold for cosine similarity between images
    """

    # Initialize Img2Vec without GPU
    img2vec = Img2Vec(cuda=False)

    # n images in directory
    n_total = len(os.listdir(path))

    # Set up dictionary to save image names (keys) and embeddings, i.e. vectors (values)
    embeddings_dictionary = {}
    imag_names = []
    for index, imag in enumerate(os.listdir(path)):
        try:
            embeddings_dictionary[imag] = img2vec.get_vec(Image.open(path + "/" + imag))
            imag_names.append(imag)
        except:
            continue

    # Compute all pairwise cosine similarities
    pairwise_similarities = defaultdict(dict)
    for image_name, vector in embeddings_dictionary.items():
        for image_name_2, vector_2 in embeddings_dictionary.items():
            if (
                image_name != image_name_2
            ):  # added this so that the same image is not compared to itself
                pairwise_similarities[image_name][image_name_2] = cosine_similarity(
                    vector, vector_2
                )

    # Get all the pairs with a cosine similarity greater than specified
    # threshold and delete the duplicates
    duplicates = []
    for image_name, vector in pairwise_similarities.items():
        for image_name_2, similarity in vector.items():
            if similarity > similarity_thesh:
                duplicates.append(
                    [image_name, image_name_2]
                )  # list of duplicate pair, append to duplicates list

    # sort the list of duplicate pairs per pair (to keep one exemplar of each
    # image)
    duplicates_sorted = [sorted(x) for x in duplicates]

    duplicates_final_A = [x[0] for x in duplicates_sorted]
    duplicates_final_B = [x[1] for x in duplicates_sorted]

    # delete only the second element of each list (using duplicates_final_B) to
    # keep one exemplar of each image
    n_deleted = 0
    for index, file in enumerate(duplicates_final_B):

        # print all ducplicate pairs
        print("Duplicates: ", file, " ", duplicates_final_A[index])

        # delete one element; some files are not deleted, this is because they
        # are already deleted in a previous iteration
        try:
            os.remove(path + "/" + file)
            print("---> deleting file: ", file)
            n_deleted = n_deleted + 1
        except:
            continue

    print("----------------------------------------------")
    print(n_deleted, " out of ", n_total, " images deleted")


def delete_small_images(
    path,
    min_width=85,
    min_height=85,
    return_del_filenames=False,
):

    """Delete small images from a directory

    Args:
        path: path to directory containing images
        min_width: minimum width of image
        min_height: minimum height of image
        return_del_filenames: if True, return list of deleted filenames

    Returns:
        file_name_deleted: list of deleted filenames
    """

    n_total = len(os.listdir(path))
    n_deleted = 0
    file_name_deleted = []
    for file in os.listdir(path):
        try:
            img = Image.open(path + "/" + file)
            width, height = img.size
            if width < min_width or height < min_height:
                os.remove(path + "/" + file)
                print("---> deleting file: ", file)
                n_deleted = n_deleted + 1
                file_name_deleted.append(file)
        except:
            continue
    print("----------------------------------------------")
    print(n_deleted, " out of ", n_total, " images deleted")
    if return_del_filenames == True:
        return file_name_deleted


def delete_extreme_aspect_ratio_images(
    path,
    max_aspect_ratio=2.0,
    min_aspect_ratio=0.5,
):
    """Delete images with extreme aspect ratio from a directory

    Args:
        path: path to directory containing images
        max_aspect_ratio: maximum aspect ratio of image
        min_aspect_ratio: minimum aspect ratio of image
    """

    n_total = len(os.listdir(path))
    n_deleted = 0
    for file in os.listdir(path):
        try:
            img = Image.open(path + "/" + file)
            width, height = img.size
            aspect_ratio = width / height
            if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
                os.remove(path + "/" + file)
                print("---> deleting file: ", file)
                n_deleted = n_deleted + 1
        except:
            continue
    print("----------------------------------------------")
    print(n_deleted, " out of ", n_total, " images deleted")


def delete_txt_files_for_del_images(
    file_names_to_delete,
    path,
    return_del_filenames=False,
):

    """Delete txt files for images that have been deleted

    Args:
        file_names_to_delete: list of file names to delete
        path: path to directory containing images
        return_del_filenames: if True, return list of deleted filenames

    Returns:
        file_name_deleted: list of deleted filenames
    """

    n_deleted = 0
    file_name_deleted = []
    for file in file_names_to_delete:
        txtfile = file.replace(".jpg", ".txt")
        os.remove(path + "/" + txtfile)
        print("---> deleting file: ", txtfile)
        n_deleted = n_deleted + 1
        file_name_deleted.append(txtfile)
    print("----------------------------------------------")
    print(n_deleted, " txt files deleted")
    if return_del_filenames == True:
        return file_name_deleted
