def polygons_to_bboxes(polygons, scale = 1):
    x_coordinates, y_coordinates = zip(*polygons)

    return [
        (min(x_coordinates) * scale, min(y_coordinates) * scale),
        (max(x_coordinates) * scale, max(y_coordinates) * scale),
    ]
