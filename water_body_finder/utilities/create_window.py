def create_window(data, point, size):
    offset = int(size/2)
    window = data[point[0]-offset: point[0] + offset,
                  point[1]-offset: point[1] + offset]
    return(window)
