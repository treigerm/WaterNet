"""Interact with the Overpass API."""

import overpass


def extract_water(bounds):
    # TODO: Test with bounding box (51.799,7.22,51.81,7.24)
    # TODO: Create cache
    # TODO: Also find relations
    api = overpass.API()
    bounding_box = "({south}, {west}, {north}, {east})".format(bounds)
    way_query = overpass.WayQuery('[natural="water"]{}'.format(bounding_box))
    response = api.Get(way_query)

    water_features = []
    elements = response['elements']
    for element in elements:
        if element['type'] == 'way':
            node_coordinates = [get_lat_lon(node_id, elements)
                                for node_id in water_features['nodes']]
            water_features.append(node_coordinates)

    return water_features


def get_lat_lon(element_id, elements):
    for element in elements:
        if element['id'] == element_id:
            return (element['lat'], element['lon'])
