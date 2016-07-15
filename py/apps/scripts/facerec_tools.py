import sys, os
sys.path.append("../..")

# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image

from facerec.feature import Fisherfaces, PCA
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.serialization import save_model, load_model

import matplotlib.cm as cm

from facerec.util import minmax_normalize

from pprint import pprint
from random import random
from datetime import datetime
import time

import cv2

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--build-model", required=False, default=None, help="Path to output file.")
ap.add_argument("-i", "--input", required=False, help="Path to the image folder of correct structure.")
ap.add_argument('-o','--output', required=False, help="Path to folder to store output model file.")
ap.add_argument("-w", "--width",required=False, default=500, type=int, help="downsizes the images before building the model.")
ap.add_argument("-c", "--components",required=False, default=0, type=int, help="number of components (for Eigenfaces (also in Fisherfaces(?). Defaults to maximum.")
ap.add_argument("-n", "--classifier-neigbours",required=False, default=1, type=int, help="this is relevant to how the predictions based on the model are returned (more detailed explanation somewhere else).")

opts = vars(ap.parse_args())



def read_images(path, sz=None):
    c = 0
    X,y = [], []
    print "\n[+] Reading images from:", path

    names = dict()

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            print subject_path
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    subject_name = subject_path.split("/")[-1]
                    y.append(c)
                    if c not in names:
                        names[c] = subject_name
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    print names
    return [X,y,names]



def computeAndSaveModel(path_to_database, path_for_model_output, size, model_type="Fisherface", num_components=0, classifier_neighbours=1):
    print "\n[+] Saving new model (confirmed below)."    
    [X,y,names] = read_images(path_to_database, sz=size)
    if model_type == "Eigenface":
        model = PredictableModel(PCA(num_components=num_components), NearestNeighbor(k=classifier_neighbours), dimensions=size, namesDict=names)
    elif model_type == "Fisherface":
        model = PredictableModel(Fisherfaces(num_components=num_components), NearestNeighbor(k=classifier_neighbours), dimensions=size, namesDict=names)
    else:
        print "[-] specify the type of model you want to comput as either 'Fisherface' or 'Eigenface' in the computeAndSaveModel function."
        return False

    model.compute(X,y)   
    save_model(path_for_model_output, model)
    print "\n[+] Saving confirmed. New model saved to:", path_for_model_output

def predictImages(path_to_img_or_folder, model):
    path = path_to_img_or_folder
    if isImage(path):
        image_name = path.split('/')[-1]
        
        im = Image.open(path)
        im = im.convert("L")
        im = im.resize(getDimensionsOfModel(model))

        img = np.asarray(im, dtype=np.uint8)
        pred = model.predict(img)
        print image_name, " ----> ", pred[1]["name"]
        print "\tfull path ----> ", path
        print "\tfull prediction ----> ", pred
        im.show()
    elif os.path.isdir(path):
        for dirname, dirnames, filenames in os.walk(path):
            print "[+] predictions for images in", dirname
            for filename in filenames:
                file_path = os.path.join(dirname, filename)

                try:
                    predictImages(file_path, model)
                except:
                    print "error, maybe no image file?"
                    pass
    else:
        print "[-] error. are you sure the path goes either to an image or a folder containing images?"



if __name__ == "__main__":

    print opts
    print opts["build_model"]

    if opts["build_model"] is not None:
        if opts["build_model"] == "Eigenfaces" or opts["build_model"] == "Fisherfaces":
            
            model_type = opts["build_model"]
            output_path = opts["output"]
            input_path = opts["input"]

            num_components = opts["components"]
            classifier_neigbours = opts["classifier_neigbours"]

            if not os.path.isdir(input_path):
                print "[+] Input directory seems to NOT EXIST:", output_path
                print "[X] Exiting."
                sys.exit()
                # os.makedirs(output_path)


            #this needs to be resolved (getting the actual dimension of the images and add warning that all images need to have the same dimensions)
            width = opts["widht"]
            height = opts["height"]

            if output_path.endswith(".pkl"):
                output_file_name = output_path.split("/")[-1]
                output_path = "/".join(output_path.split("/")[:-1])
            else:
                output_file_name = "model.plk"


            if not os.path.isdir(output_path):
                print "[+] Creating directories:", output_path
                os.makedirs(output_path)

            output_path = os.path.join(output_path, output_file_name)

            print "[+] Building a", model_type, "model."
            print '\toutput_path:', output_path
            print '\tinput_path:', input_path

            print '\tcomponents', num_components
            print '\tclassifier_neigbours', classifier_neigbours
            print '\tdimensions', width, "x", height



            # computeAndSaveModel(input_path, 'model.pkl', size=(size,size), model_type="Eigenface", num_components=0, classifier_neighbours = 1)

    # if len(sys.argv) > 1:
    #     path_to_database = sys.argv[1]
    #     # if not os.path.isdir(path_to_database):
    #     #     print "Wrong path to database provided / folder doesn't exist."
    #     #     sys.exit()

    #     size = 800

    #     computeAndSaveModel(path_to_database, 'model.pkl', size=(size,size), model_type="Eigenface", num_components=0, classifier_neighbours = 1)

    #     file_path = path_file_from(path_to_database)

    #     model = loadModel('model.pkl')

    #     # showModel(model, colormap=cm.gray)

    #     # predictImages(path_to_database, model)
    #     predictOptimize(file_path, model, size)

        # trevor_face = "../../../../facerec/data/tp_aligned_cropped/tiff/_B9A3986.JPG_frame_00000001_out.tiff"
        # reconstructFaceFromModel("/HTSLAM/input/Leon/face_dataset/ck_formatted/S011/S011-105.tiff", model)

        # showModel(model, colormap=cm.gray)







