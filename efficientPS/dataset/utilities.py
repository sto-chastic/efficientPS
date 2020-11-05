def polygons_to_bboxes(polygons, offset = [0, 0]):
    x_coordinates, y_coordinates = zip(*polygons)

    return [
        (min(x_coordinates) - offset[0], min(y_coordinates) - offset[1]),
        (max(x_coordinates) - offset[0], max(y_coordinates) - offset[1]),
    ]
