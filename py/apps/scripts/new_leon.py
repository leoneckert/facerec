import sys, os
sys.path.append("../..")

# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image

from facerec.feature import Fisherfaces
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
    print "\n\t[i] Index-references of subjects:\n"
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)

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
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            print "\t\t", c, "refers to", subject_name
            c = c+1
    return [X,y]




def computeAndSaveModel(path_to_database, path_for_model_output, size):
    print "\n[+] Saving new model (confirmed below)."    
    [X,y] = read_images(path_to_database, sz=size)
    model = PredictableModel(Fisherfaces(), NearestNeighbor(), dimensions=size)
    model.compute(X,y)   
    save_model(path_for_model_output, model)
    print "\n[+] Saving confirmed. New model saved to:", path_for_model_output

def loadModel(path_to_model):
    print "\n[+] Loading model from:", path_to_model
    return load_model(path_to_model)


def getDimensionsOfModel(model):
     return model.dimensions

def showFisherfaces(model, colormap=None):
    """
    Opens Fisherfaces of a given model.
    """

    print "\n[+] Creating Fisherfaces for model:", model
    dimensions = getDimensionsOfModel(model)
    E = []
    print ""
    for i in xrange(min(model.feature.eigenvectors.shape[1], 20)):
        e = model.feature.eigenvectors[:,i].reshape((dimensions))
        e = minmax_normalize(e,0,255, dtype=np.uint8)
        if colormap is None:
            img = Image.fromarray(e)
        else:
            img = Image.fromarray(np.uint8(colormap(e)*255))
        print "\t[o] Opening Fisherfaces [" + str(i) + "]"
        img.show()



if __name__ == "__main__":

    

    if len(sys.argv) > 1:
        path_to_database = sys.argv[1]
        if not os.path.isdir(path_to_database):
            print "Wrong path to database provided / folder doesn't exist."
            sys.exit()


    computeAndSaveModel(path_to_database, 'model.pkl', size=(700,700))

    model = loadModel('model.pkl')

    showFisherfaces(model, colormap=cm.gray)

