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

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    print "\n[+] Reading images from:", path
    # print "\n\t[i] Index-references of subjects:\n"

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
            # print "\t\t", c, "refers to", subject_name
            c = c+1
    print names
    return [X,y,names]




def computeAndSaveModel(path_to_database, path_for_model_output, size, model_type="Fisherface", num_components=0):
    print "\n[+] Saving new model (confirmed below)."    
    [X,y,names] = read_images(path_to_database, sz=size)
    if model_type == "Eigenface":
        model = PredictableModel(PCA(num_components=num_components), NearestNeighbor(), dimensions=size, namesDict=names)
    elif model_type == "Fisherface":
        model = PredictableModel(Fisherfaces(num_components=num_components), NearestNeighbor(), dimensions=size, namesDict=names)
    else:
        print "[-] specify the type of model you want to comput as either 'Fisherface' or 'Eigenface' in the computeAndSaveModel function."
        return False

    model.compute(X,y)   
    save_model(path_for_model_output, model)
    print "\n[+] Saving confirmed. New model saved to:", path_for_model_output

def loadModel(path_to_model):
    print "\n[+] Loading model from:", path_to_model
    return load_model(path_to_model)


def getDimensionsOfModel(model):
     return model.dimensions

def showModel(model, colormap=None):
    """
    Opens Fisherfaces of a given model.
    """

    print "\n[+] Creating Fisherfaces for model:", model
    dimensions = getDimensionsOfModel(model)
    E = []
    print ""
    for i in xrange(min(model.feature.eigenvectors.shape[1], 20)):
        print model.feature.eigenvectors[:,i]
        e = model.feature.eigenvectors[:,i].reshape((dimensions))

        e = minmax_normalize(e,0,255, dtype=np.uint8)
        if colormap is None:
            img = Image.fromarray(e)
        else:
            img = Image.fromarray(np.uint8(colormap(e)*255))
        print "\t[o] Opening Fisherfaces [" + str(i) + "]"
        img.show()

def isImage(path):
    # potentially more sophisticated, checking file format etc.
    if os.path.isdir(path):
        return False
    elif os.path.isfile(path):
        return True


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


def reconstructFaceFromModel(path_to_input_image, model, save_path = None):
    im = Image.open(path_to_input_image)
    im = im.convert("L")
    im = im.resize(getDimensionsOfModel(model), Image.ANTIALIAS)
    img = np.asarray(im, dtype=np.uint8)
    ex = model.feature.extract(img)
    re = model.feature.reconstruct(ex)
    re = re.reshape(getDimensionsOfModel(model))
    e = minmax_normalize(re,0,255, dtype=np.uint8)
    img = Image.fromarray(e)
    if save_path == None:
        img.show()
    else:
        img.save(save_path)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path_to_database = sys.argv[1]
        # if not os.path.isdir(path_to_database):
        #     print "Wrong path to database provided / folder doesn't exist."
        #     sys.exit()


        # computeAndSaveModel(path_to_database, 'model.pkl', size=(300,300), model_type="Eigenface", num_components=0)


        model = loadModel('model_cohn_kanade_300_300.pkl')


        # predictImages(path_to_database, model)

        # trevor_face = "../../../../facerec/data/tp_aligned_cropped/tiff/_B9A3986.JPG_frame_00000001_out.tiff"
        reconstructFaceFromModel("/HTSLAM/input/Leon/face_dataset/ck_formatted/S011/S011-105.tiff", model)

        # showModel(model, colormap=cm.gray)








    # CREATE A FULL SEQUENCE (TAKES A LONG TIME)


    # for i in range(1, 135):
    #     num_c = i * 6  
        
    #     computeAndSaveModel(path_to_database, 'model.pkl', size=(300,300), model_type="Eigenface", num_components=num_c)


    #     model = loadModel('model.pkl')


    #     # predictImages(path_to_database, model)

    #     trevor_face = "../../../../facerec/data/tp_reduced/trevor/_B9A4018.JPG_frame_00000001_out.tiff"
    #     reconstructFaceFromModel(trevor_face, model, "/HTSLAM/output/Leon/cohn_kanade_sequence/trevor/trevor_ck_" + str(num_c) + ".jpg")

    #     # showModel(model, colormap=cm.gray)

