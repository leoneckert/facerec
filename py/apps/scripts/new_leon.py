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
from time import time

import cv2

def print_progress(count, total=0, label="Progress:"):
    '''
    prints progress on the same line.
    req: import sys
    '''
    if total is 0:
        to_print =  "\r" + label + str(count).rjust(len(str(count))+1)
    elif total > 0:
        
        loadingbar_length = 40
        num_bars = int(float(loadingbar_length)*(float(count)/float(total)))
        # loadingbar = "|".rjust(3) +"="*num_bars+">"+   (  ("| |".rjust(loadingbar_length - num_bars + 3)) if (float(count)/float(total)) < 1 else ("|X|".rjust(loadingbar_length - num_bars + 3))      )
        the_x = '|' + '\x1b[%sm%s\x1b[0m' % ('0;32;40', 'X') + '|'
        loadingbar = "|".rjust(3) +"="*num_bars+">"+   (  ("| |".rjust(loadingbar_length - num_bars + 3)) if (float(count)/float(total)) < 1 else (the_x.rjust(loadingbar_length - num_bars + 3))      )
        percentage = str(float(count)/float(total)*100)[:6].rjust(7) + "%" 

        count_vs_total = str(count).rjust(len(str(total))+1) + "/" + str(total) 
        
        to_print =  "\r" + label + count_vs_total + loadingbar + percentage

    sys.stdout.write(to_print)
    sys.stdout.flush()
def stop_print_progress():
    print ""  

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

def predictOptimize(path_to_img_or_folder, model, size, predictTime = False):
    path = path_to_img_or_folder
    
    out_path = '/HTSLAM/output/Leon/noise_to_face/ntf_test_'
    c = 0
    new_path = out_path + str(c)
    while os.path.isdir(new_path):
        c += 1
        new_path = out_path + str(c)
    os.mkdir(new_path)




    im = Image.open(path)
    im = im.convert("L")
    im = im.resize(getDimensionsOfModel(model))



    im.save(new_path + "/from_orig_db.jpg")

    pix = im.load()
    # start by setting all pixels blank:
    for i in range(getDimensionsOfModel(model)[0]):
        for j in range(getDimensionsOfModel(model)[1]):
            pix[i,j] = 255

    time_factor = 8000 # the smaller the more rectangles for each size (always exponentially more for the smaller they get)
    length_interval = 2 # defines how many pixels the size of rectangle increases in each step
    current_distance = 100000000


    pixel_analysed = 0
    num_predictions = 0
    pixel_drawn = 0
    l_count = 0
    totaldraw = 0
    totalanalysed = 0
    totalpred = 0
    for i in range(size/length_interval):
        loops_per_size = ((1+i*length_interval)*(1+i*length_interval))/time_factor
        numpredictions = loops_per_size * 255
        pixel_drawn = (((size+1)-(i*length_interval))*((size+1)-(i*length_interval)))*loops_per_size
        pixel_analysed = pixel_drawn*255
        
        totaldraw += pixel_drawn
        totalanalysed += pixel_analysed
        totalpred += numpredictions
    
    loop_secs = float(size/length_interval) * 0.205318021774
    print "total loops:", i, loop_secs, "seconds"   
    
    draw_secs = float(totaldraw) * 2.25419623148e-07
    print "total pixels drawn:", totaldraw, draw_secs, "seconds"
    
    analyse_secs = float(totalanalysed) * 2.25419623148e-07
    print "total pixels analysed:", totalanalysed, analyse_secs, "seconds"
    
    pred_secs = float(totalpred) * 0.00122822872827
    print "total pixels predicted:", totalpred, pred_secs, "seconds" # maybe rather 0.000788020398718

    pred_time_total = loop_secs + draw_secs + analyse_secs + pred_secs
    print "predicted time:", pred_time_total, "seconds"
    print "\t\t", pred_time_total/60.0, 'minutes'

    # testiing predicions: 
    # time_factor = 7 length_interval = 2, size = 30, predicted = 207.208512531, actual = 131.811971188
    # time_factor = 2 length_interval = 1, size = 30, predicted = 1516.3374941,  actual = 967.648044109
    # time_factor = 1000 length_interval = 10, size = 700, predicted = 36000.228859, actual = 21932.0193892
    # time_factor = 50000 length_interval = 2, size = 700, predicted = 2571.9817542, actual = 1999.86883903


    img = np.asarray(im, dtype=np.uint8)
    timepre = time()
    for length_fraction in range(size/length_interval):

        length_n = length_fraction * length_interval

        length = (size+1) - length_n # current length
        t = 0

        loops_per_size = ((1+length_n)*(1+length_n))/time_factor
        num_predictions = 0
        for i, times in enumerate( range(loops_per_size), 1):
            t = times
            print_progress(i, loops_per_size, "[+] "+str(length_n + 1)+"/"+str(size) +" - Currently optimizing rectangels with size 0-" + str(length) + ":")
 
            w = int(random()*length)
            h = int(random()*length)
            ran_x = int(random()*(size-w))
            ran_y = int(random()*(size-h))
            rect_start = (ran_x,ran_y)
            rect_end = (ran_x+w,ran_y+h)

            mini = 100000000
            mini_b = -5
            for b in range(255):

                cv2.rectangle(img, rect_start, rect_end, b, -1)

                pred = model.predict(img)

                dist = pred[1]['distances'][0]
                if dist < mini:
                    mini = dist
                    mini_b = b

            current_distance = mini

            cv2.rectangle(img, rect_start, rect_end, mini_b, -1)

        
        print "\t-current distance:", current_distance
        cv2.imwrite(new_path + "/face_length_" + str(length) + "x" + str(t) + ".jpg", img)
    
    print "total time:", time() - timepre, "seconds"
        


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

def path_file_from(path_to_database):
    for dirname, dirnames, filenames in os.walk(path_to_database):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                return os.path.join(subject_path, filename)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        path_to_database = sys.argv[1]
        # if not os.path.isdir(path_to_database):
        #     print "Wrong path to database provided / folder doesn't exist."
        #     sys.exit()

        size = 700

        computeAndSaveModel(path_to_database, 'model.pkl', size=(size,size), model_type="Eigenface", num_components=0, classifier_neighbours = 1)

        file_path = path_file_from(path_to_database)

        model = loadModel('model.pkl')

        # showModel(model, colormap=cm.gray)

        # predictImages(path_to_database, model)
        predictOptimize(file_path, model, size)

        # trevor_face = "../../../../facerec/data/tp_aligned_cropped/tiff/_B9A3986.JPG_frame_00000001_out.tiff"
        # reconstructFaceFromModel("/HTSLAM/input/Leon/face_dataset/ck_formatted/S011/S011-105.tiff", model)

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
















"""
prediction optimisation thoughts:

___________
Eigenfaces:

when constructing a eigenface model using only two input images, the calculation is much faster, but what's more interesting is
that the distance is small from the start, then very small, very quickly and then hardly improving/more arbitrary.

when using many input photos to construct a 'larger' eigenface model, computation is slower, 
but the distances have more space to actually improve - seems more reliable, the look seems to actually *matter* with this method.

____________
Fisherfaces:

FIsherface not so much explored yet. One way I can think if is to check WHO is predicted and then, if the WHO is
the target, how low the distance. If, for no value the target is predicted, don't draw the rectangle.

"""