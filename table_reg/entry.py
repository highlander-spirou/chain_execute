import networkx as nx
from image_processor import *
from index import add_functions_to_graph, ChainExecutor, create_edge
import cv2

Graph = nx.DiGraph()
add_functions_to_graph(Graph, [convert_image_to_grayscale, convert_grayscale_to_blacknwhite, invert_image,
                       dilate_image, find_contours, find_rectangles, find_boundaries, order_boundaries])


create_edge(Graph, 'convert_image_to_grayscale', 'convert_grayscale_to_blacknwhite')
create_edge(Graph, 'convert_grayscale_to_blacknwhite', 'invert_image')
create_edge(Graph, 'invert_image', 'dilate_image')
create_edge(Graph, 'dilate_image', 'find_contours')
create_edge(Graph, 'find_contours', 'find_rectangles')
create_edge(Graph, 'find_rectangles', 'find_boundaries')
create_edge(Graph, 'find_boundaries', 'order_boundaries')

if __name__ == "__main__":
    img = cv2.imread('./imgs/test_3.jpg')


    grayscale_img = convert_image_to_grayscale(img)
    blackwhite_img = convert_grayscale_to_blacknwhite(grayscale_img)
    inverted_img = invert_image(blackwhite_img)
    dilated_img = dilate_image(inverted_img)
    contours = find_contours(dilated_img)
    rectangles = find_rectangles(contours)
    boundaries = find_boundaries(rectangles)
    # ordered_boundaries = order_boundaries(boundaries)
    img_ = img.copy()
    output = cv2.drawContours(img_, [boundaries], -1, (0, 255, 0), 3)

    # print(len(contours))


    # executor = ChainExecutor(Graph)
    # executor.set_external_dependencies('transform_grayscale', img)
    # executor.execute('order_boundaries')
    # data = executor.get_node_data('dilate_image')
    # output = draw_boundaries(img, data)
    cv2.imwrite('./output/test_2.png', output)