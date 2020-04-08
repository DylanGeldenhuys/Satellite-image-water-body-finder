import numpy as np


def order_edge_points(complex_points):
    complex_points_ls = list(complex_points)
    ordered_list = [[]]
    polygon_index = 0
    point = complex_points_ls[0]
    point_index = 0

    while len(complex_points_ls) > 1:
        point_found = False
        for i in range(len(complex_points_ls)):
            if (abs(complex_points_ls[i][0] - point[0]) <= 1 and abs(complex_points_ls[i][1] - point[1]) <= 1):
                if ((point[0], point[1]) != (complex_points[i][0], complex_points[i][1])):
                    ordered_list[polygon_index].append(list(point))
                    point = complex_points_ls[i]
                    complex_points_ls.pop(point_index)
                    point_index = i
                    point_found = True
                    break

        if point_found == False:
            ordered_list[polygon_index].append(list(point))
            complex_points_ls.pop(point_index)
            point = complex_points_ls[0]
            point_index = 0
            polygon_index += 1
            ordered_list.append([])

    ordered_list[polygon_index].append(list(point))

    return ordered_list


def get_boundary(mask):
    edges = []

    height = mask.shape[0] - 1
    width = mask.shape[1] - 1

    for j in range(1, height):
        for i in range(1, width):
            if (mask[j, i] == 0):
                if (mask[j, i + 1] == 1 or mask[j, i - 1] == 1 or mask[j + 1, i] == 1 or mask[j - 1, i] == 1):
                    edges.append([j, i])
    return edges
