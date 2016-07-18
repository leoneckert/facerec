import cv2
import numpy as np
# from random
from random import random, choice
from PIL import Image, ImageDraw


class  PolygonElement:
    """a polygon"""
    
    def __init__(self, dimensions, black_and_white = True):
        self.num_vertices = 3
        self.width = dimensions[0]
        self.height = dimensions[1]
        self.points = self.init_points()
        self.alpha = int(random()*255)
        self.black_and_white = black_and_white
        self.color = self.init_color()
        self.active = False

        # self.config_backup_copy = self.get_config()
        self.currently_modified_config = None

    def init_vertex(self):
        return ( int(random()*self.width), int(random()*self.height) )

    def init_points(self):
        points = list()
        for i in range(self.num_vertices):
            (x, y) = self.init_vertex()
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

        categories = ["points", "alpha", "color"]
        cat_to_change = choice(categories)

        if cat_to_change == "color":
            self.currently_modified_config["color"] = self.init_color()
        elif cat_to_change == "alpha":
            self.currently_modified_config["alpha"] = int(random()*255)
        elif cat_to_change == "points":
            ran_idx = int(random()*self.num_vertices)
            self.currently_modified_config["points"][ran_idx] = self.init_vertex()

        return self.draw_from_config(canvas, config=self.currently_modified_config, temp_test=True)

    def apply_modified_config(self):
        print "Lets apply this",
        c = self.currently_modified_config
        print c
        self.num_vertices = c["num_vertices"]
        self.width = c["width"]
        self.height = c["height"]
        self.points = c["points"]
        self.alpha = c["alpha"]
        self.black_and_white = c["black_and_white"]
        self.color = c["color"]
        self.active = True


        # self.active = True




class ElementOrganizer:
    """organizer object"""
    def __init__(self, canvas, num_elements = 10,  black_and_white = True):
        self.canvas = canvas
        self.elements = self.init_elements(num_elements, black_and_white)
        self.num = num_elements
        self.distance = None
        

    def init_elements(self, num_elements, black_and_white):
        elems = list()
        for i in range(num_elements):
            elems.append(  PolygonElement(self.canvas.size, black_and_white=black_and_white)  )       
        return elems

    def prediction_placeholder(self, canvas):
        return int(random()*20)

    def progress_attempt(self):
        ran_elem_idx = int(random()*self.num)

        temp_canvas = self.canvas.copy()

        temp_canvas = self.elements[ran_elem_idx].apply_random_change_to_temp_canvas(temp_canvas)

        new_distance = self.prediction_placeholder(temp_canvas) 

        if self.distance == None or new_distance < self.distance:
            # save temporary settings in element
            self.elements[ran_elem_idx].apply_modified_config()
            # update main canvas
            self.canvas = temp_canvas
        #     # update main distance
            print "old distance:", self.distance
            self.distance = new_distance
            print "new distance:", self.distance
            self.canvas.show()
        # else: 
        #     # reset temporary settings in element (not sure if even needed)


        



    # def draw_all(self, base):
    #     for element in self.elements:
    #         base = element.draw(base)
    #     return base

    def run(self):
        while True:
            self.progress_attempt()





if __name__ == '__main__':

    width = 300
    height = 300

    canvas = Image.new("RGBA", (width,height), (255,255,255,255))


    # p = PolygonElement(canvas.size)
    
    # im = p.draw(canvas)
    

    # for i in range(100):
    #     p2 = PolygonElement(canvas.size)    
    #     im = p2.draw(im)
    # im.show()

    organizer = ElementOrganizer(canvas, num_elements = 2,  black_and_white = True)
    
    organizer.run()

    # im = organizer.draw_all(canvas)
    # im.show()
