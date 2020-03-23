

# Imports

import numpy as np 


# feature functions
def extract_variance(window):
    data = window[:, :, 0]

    shape = data.shape
    return(np.var(data[0:shape[0], 0:shape[1]].flatten()))
    
def exract_colour(window):
    return(window.mean(axis=0).mean(axis=0))
    # returns shape (3,)

def extract_label(mask_window):
    mean = mask_window.mean(axis=0).mean(axis=0)[0]
    return(mean < 127)

# utility functions

def create_window(image,point,size):
    off_centre = round(size/2)
    window = image[ point[0]-off_centre : point[0] + off_centre, point[1]-off_centre : point[1] + off_centre]
    return(window)

# Main

# gets single point sample
def create_training_sample(point, size, image, label, image_name, second_window_factor):
    window_1 = create_window(image,point,size)
    window_2 = create_window(image,point,size*second_window_factor)

    variance = extract_variance(window_1)
    color = exract_colour(window_1)
    variance2 = extract_variance(window_2)
    color2 = exract_colour(window_2)


    sample = [image_name, variance,color[0],color[1],color[2],variance2,color2[0],color2[1],color2[2], label]
    return sample

# gets samples for whole image
def create_training_samples(size, image, mask, image_name):
    second_window_factor = 2

    height = int(len(image) / size) - second_window_factor
    width = int(len(image[1]) / size) - second_window_factor

    data_set = []

    offset = round(size/2)

    yes = 0
    no = 0

    for j in range(height):
        for i in range(width):
            y = j * size + offset
            x = i * size + offset
            
            label_window = create_window(mask, [x,y], size)
            label = extract_label(label_window)

            if (label):
                yes += 1
                data_set.append(create_training_sample([x,y], size, image, label, image_name, second_window_factor))
            elif(yes >= no):
                no += 1
                data_set.append(create_training_sample([x,y], size, image, label, image_name, second_window_factor))

    return(data_set)