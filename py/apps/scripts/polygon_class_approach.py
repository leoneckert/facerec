import cv2
import sys, os
import argparse
import numpy as np
import math
# from random
from random import random, choice, sample
from PIL import Image, ImageDraw

sys.path.append("../..")
from facerec.feature import Fisherfaces, PCA
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.serialization import save_model, load_model

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-path", required=False, default=None, help="Path to output file.")
opts = vars(ap.parse_args())

def print_progress(count, total=0, label="Progress:", _loadingbar_length = 40):
    '''
    prints progress on the same line.
    req: import sys
    '''
    to_print =  "\r" + label + str(count).rjust(len(str(count))+1)
    sys.stdout.write(to_print)
    sys.stdout.flush()

def stop_print_progress():
    # print ""
    to_print =  "\r"
    sys.stdout.write(to_print)
    sys.stdout.flush()

class  PolygonElement:
    """a polygon"""
    
    def __init__(self, dimensions, index, alpha=80, max_size=None, black_and_white = True):
        self.index = index
        self.num_vertices = 5
        self.width = dimensions[0]
        self.height = dimensions[1]
        if max_size is None:
            # this is assuming we always draw on a square canvas
            self.max_size = dimensions[0]
        else:
            self.max_size = max_size
        self.points = self.init_points()
        # self.alpha = int(random()*255)
        # self.alpha = 40\
        self.alpha = alpha
        self.black_and_white = black_and_white
        self.color = self.init_color()
        self.active = False

        self.currently_modified_config = None

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )


    def init_vertex(self, ref_point):
        # print ref_point
        # new_vertex = ( int(random()*self.width), int(random()*self.height) )
        if ref_point is not None:
            new_vertex = ( ref_point[0] + int( (random()*2-1)*self.max_size  ) , ref_point[1] + int( (random()*2-1)*self.max_size ))
            while new_vertex[0] > self.width or new_vertex[0] < 0 or new_vertex[1] > self.height or new_vertex[1] < 0:
                new_vertex = ( ref_point[0] + int( (random()*2-1)*self.max_size  ) , ref_point[1] + int( (random()*2-1)*self.max_size ))
        else:
            new_vertex = ( int(random()*self.width), int(random()*self.height) )
            # while self.distance(new_vertex, ref_point) > self.max_size:
            #     new_vertex = ( int(random()*self.width), int(random()*self.height) )
            # print "new_vertex", new_vertex, "ref_points", ref_point, "distance", self.distance(new_vertex, ref_point)
        return new_vertex

    def init_points(self):
        points = list()
        ref_points = None
        for i in range(self.num_vertices):
            (x, y) = self.init_vertex(ref_points)
            if i == 0: ref_points = (x,y)
            points.append( (x,y) )
        return points

    def init_color(self):
        if self.black_and_white is True:
            c = int(random()*255)
            return (c,c,c)
        return ( int(random()*255),int(random()*255),int(random()*255) )

    def get_config(self):
        config = {
            "num_vertices": self.num_vertices,
            "width": self.width,
            "height": self.height,
            "points": self.points,
            "alpha": self.alpha,
            "black_and_white": self.black_and_white,
            "color": self.color,
            "active": self.active
        }
        return config

    def draw_from_config(self, base, config=None, temp_test = False):
        if config is None:
            config = self.get_config()

        if config["active"] is True or temp_test is True:
            im = Image.new("RGBA", base.size, (255,255,255,0))
            draw = ImageDraw.Draw(im)
            color = ( config["color"][0], config["color"][1], config["color"][2], config["alpha"])
            draw.polygon(config["points"], fill = color)      
            return Image.alpha_composite(base, im)
        return base

    def apply_random_change_to_temp_canvas(self, canvas):
        self.currently_modified_config = self.get_config()

        categories = ["points", "color", "point"]
        cat_to_change = choice(categories)

        if cat_to_change == "color":
            self.currently_modified_config["color"] = self.init_color()
        elif cat_to_change == "alpha":
            self.currently_modified_config["alpha"] = int(random()*255)
        elif cat_to_change == "points":
            self.currently_modified_config["points"] = self.init_points()
        elif cat_to_change == "point":
            ran_points = sample(range(self.num_vertices), 2)
            ran_idx = ran_points[0]
            ran_ref_idx = ran_points[1]
            ran_ref = self.currently_modified_config["points"][ran_ref_idx]
            self.currently_modified_config["points"][ran_idx] = self.init_vertex(ran_ref)

        return self.draw_from_config(canvas, config=self.currently_modified_config, temp_test=True)

    def apply_modified_config(self):

        c = self.currently_modified_config

        self.num_vertices = c["num_vertices"]
        self.width = c["width"]
        self.height = c["height"]
        self.points = c["points"]
        self.alpha = c["alpha"]
        self.black_and_white = c["black_and_white"]
        self.color = c["color"]
        self.active = True



def getDimensionsOfModel(model):
     return model.dimensions

class ElementOrganizer:
    """organizer object"""
    def __init__(self, canvas,  model_path=None, num_elements = 10,  black_and_white = True):
        self.canvas = canvas
        self.blank_canvas = canvas
        self.elements = self.init_elements(num_elements, black_and_white)
        self.num = num_elements
        self.distance = None
        self.model = load_model(model_path)
        self.improve_count = 0
        self.tries_since_last_improvement = 0
        print getDimensionsOfModel(self.model)

    def init_elements(self, num_elements, black_and_white):
        elems = list()
        for i in range(num_elements):
            # assuming square canvas again
            maximum_size = int( self.canvas.size[0] - ((float(i)/num_elements)*self.canvas.size[0]) )
            if maximum_size < 1: maximum_size = 1
            alpha = 20
            print "[+] polygon with max size:", maximum_size, "alpha:", alpha
            # maximum_size = 30
            elems.append(  PolygonElement(self.canvas.size, i, alpha=alpha, max_size= maximum_size ,black_and_white=black_and_white)  )       
        return elems

    def prediction_placeholder(self, canvas):
        return int(random()*30)

    def draw_all(self, modify_index=None):
        base = self.blank_canvas
        for i in range(self.num):
            if i == modify_index:
                base =  self.elements[i].apply_random_change_to_temp_canvas(base)
            else:
                base = self.elements[i].draw_from_config(base)
        return base


    def progress_attempt(self):
        ran_elem_idx = int(random()*self.num)

        temp_canvas = self.canvas.copy()

        # temp_canvas = self.elements[ran_elem_idx].apply_random_change_to_temp_canvas(temp_canvas)
        temp_canvas = self.draw_all(modify_index = ran_elem_idx)
        # temp_canvas.show()
        
        # new_distance = self.prediction_placeholder(temp_canvas) 
        
        im = temp_canvas.convert("L")
        img = np.asarray(im, dtype=np.uint8)
        pred = self.model.predict(img)
        # print pred[1]['labels']
        # print pred[1]['name']
        valid = False
        if pred[1]['name'] == 'Hito':
        # if 2 in pred[1]['labels']:
            # print pred[1]['name']
            valid = True
        else:
            print "wrong subject"

        new_distance = pred[1]['distances'][0]
        # print new_distance, self.distance
        print_progress(self.tries_since_last_improvement, label="Testing modifications:")
        if (valid and self.distance == None) or (valid and new_distance < self.distance):
            stop_print_progress() 
            self.improve_count += 1
        #     # save temporary settings in element
            self.elements[ran_elem_idx].apply_modified_config()
        #     # update main canvas
            self.canvas = self.draw_all()
        # #     # update main distance
            # print "old distance:", self.distance
            self.distance = new_distance
            print "[+] Improvement "+str(self.improve_count)+" | New distance:", self.distance,"| tries taken:", self.tries_since_last_improvement
            self.tries_since_last_improvement = 0
            # self.canvas.show()

            self.canvas.save("/HTSLAM/output/Leon/polygon_eigenface_hallucination/2/frame_"+str(self.improve_count).zfill(7)+".jpg")
        else: 
            self.tries_since_last_improvement += 1
        #     # reset temporary settings in element (not sure if even needed)


        



    # def draw_all(self, base):
    #     for element in self.elements:
    #         base = element.draw(base)
    #     return base

    def run(self):
        while True:
            self.progress_attempt()
        # self.draw_all()




if __name__ == '__main__':
    print opts
    model_path = opts["model_path"]
    if not os.path.isfile(model_path) or not model_path.endswith(".plk"):
        print "[+] Not a valid model file:", output_path
        print "[X] Exiting."
        sys.exit()

    model = load_model(model_path)


    width = getDimensionsOfModel(model)[0]
    height = getDimensionsOfModel(model)[1]

    canvas = Image.new("RGBA", (width,height), (255,255,255,255))


    # p = PolygonElement(canvas.size)
    
    # im = p.draw(canvas)
    

    # for i in range(100):
    #     p2 = PolygonElement(canvas.size)    
    #     im = p2.draw(im)
    # im.show()

    organizer = ElementOrganizer(canvas, model_path=model_path, num_elements = 50000,  black_and_white = True)
    
    organizer.run()

    # im = organizer.draw_all(canvas)
    # im.show()
