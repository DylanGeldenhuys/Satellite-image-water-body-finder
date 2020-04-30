#Authors: Simon, Dylan

def get_boundary(result):
    edges = []
    for j in range(result.shape[0] - 1):
        for i in range(result.shape[1] - 1):
            if (result[j, i] == True):
                if (result[j, i] != result[j, i + 1] or result[j, i] != result[j, i - 1] or result[j, i] != result[j + 1, i] or result[j, i] != result[j - 1, i]):
                    edges.append([j,i])
    return(edges)

def get_boundary2(mask):
    edges = []
    for j in range(mask.shape[0]-1):
        for i in range(mask.shape[1]-1):
            if (mask[j, i] == True):
                if (mask[j, i + 1] == False
                or mask[j, i - 1] == False
                or mask[j + 1, i] == False
                or mask[j - 1, i] == False):
                    edges.append([j, i])
    return(edges)


def order_points_simply(complex_points):
    complex_points_ls = list(complex_points)
    simple_lists = [[]]
    polygon_index = 0
    point = complex_points_ls[0]
    point_index = 0
    while len(complex_points_ls) > 1:
        point_found = False
        #print(len(complex_points_ls))
        for i in range(len(complex_points_ls)):
            if (abs(complex_points_ls[i][0] - point[0]) <= 1 and abs(complex_points_ls[i][1] - point[1]) <= 1):
                if (point[0],point[1]) != (complex_points_ls[i][0],complex_points_ls[i][1]):
                    simple_lists[polygon_index].append(list(point))
                    point = complex_points_ls[i]
                    complex_points_ls.pop(point_index)
                    if point_index < i:
                        point_index = i-1
                    else:
                        point_index = i
                    point_found = True
                    break
        if point_found == False:
            simple_lists[polygon_index].append(list(point))
            complex_points_ls.pop(point_index)
            point = complex_points_ls[0]
            point_index = 0
            polygon_index += 1
            simple_lists.append([])
    simple_lists[polygon_index].append(list(point))        
    return simple_lists
