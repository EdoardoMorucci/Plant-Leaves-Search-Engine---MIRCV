import os
import time
from statistics import median
import numpy as np
import pandas as pd
import random
import math
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import io_ops, image_ops

from PlantLeavesSearchEngine import settings
from .VantagePointTree import Object
from .VantagePointTree import Node
from .VantagePointTree import VantagePointTree



def extract_features(query_path, model):
  image_width = 224
  image_height = 224
  image_size = (image_height, image_width)

  image = io_ops.read_file(query_path)
  image = image_ops.decode_image(image, channels=3, expand_animations=False)
  image = image_ops.resize_images_v2(image, image_size, method='bilinear')
  image.set_shape((224, 224, 3))
  image = image.numpy()
  image = np.array([image])
  query_features = model(image)
  return query_features

def initalize():
    global dataframe
    global vp_tree_index
    global fixed_model
    dataframe = pd.read_csv("./media/dataframe.csv")
    with open(os.path.join(settings.MEDIA_ROOT, "Index.pkl"), 'rb') as f:
        loaded_dict = pickle.Unpickler(f).load()
    f.close()
    new_root = Node.from_dict(loaded_dict)
    vp_tree_index = VantagePointTree(root=new_root, size=97034, from_disk=True, distance_measure="euclidian")

    model = keras.models.load_model('./media/model_fine_tuned')
    fixed_model = keras.models.Model(inputs=model.input, outputs=model.get_layer('gap').output)


def search(filename):


    query_features = extract_features(os.path.join(settings.MEDIA_ROOT, filename), fixed_model)

    print(query_features)

    k = 10
    query_object = Object(query_features.numpy(), 0)
    start = time.time()
    kNN, dNN, distance_computations = vp_tree_index.search_kNN(query_object, k)
    end = time.time() - start

    print(f"time:{end:.2f}")

    neig = []

    for i in range(k):
        neig.append(kNN[i].id)

    print(f"Ho computato {distance_computations} su {vp_tree_index.size} oggetti nell'indice")


    # devo trovare i path tramite il dataframe devo levare i primi 9 caratteri

    image_paths = dataframe.Path.iloc[neig]
    plant_types = dataframe.Plant.iloc[neig]
    correct_paths = []

    for path in image_paths:
        correct_paths.append(path[8:])

    return correct_paths, plant_types
