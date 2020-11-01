def polygons_to_bboxes(polygons):
    x_coordinates, y_coordinates = zip(*polygons)

    return [
        (min(x_coordinates), min(y_coordinates)),
        (max(x_coordinates), max(y_coordinates)),
    ]
