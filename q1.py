import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def event_handler(event, x, y, flags, param):
    """
    Event handler for opencv setMouseCallback
    :param event: mouse click
    :param x: x coordinate of chosen point
    :param y: y coordinate of chosen point
    :param flags: not used. but it's necessary
    :param param: the list to append x,y into
    :return: None
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([y, x])
        print(y, x)
    else:
        pass


def get_user_points(input_image, img_name):
    """
    Shows the input image so that the user can input a point on sample birds.
    :param input_image: image to be shown
    :param img_name: directory of the image
    :return: list of user points.
    """
    user_points = []
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(img_name, event_handler, param=user_points)
    while True:
        cv2.imshow(img_name, input_image)
        # Use Esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    # user_points = np.array(user_points)
    return user_points


def points_file_reader(file_name):
    """
    This function takes the name of the file containing coordinates of the points and returns the points as an array.
    :param file_name: name of the file containing coordinates of the points
    :return: a numeric array containing coordinates of the points
    """
    with open(file_name, 'r') as f:
        points_str = f.readlines()  # Reads all the lines at once
    # The first line is the number of points:
    n_points = points_str[0]
    # Remove the next line character
    n_points = int(n_points[:-1])
    # Separate coordinates by space and assign store them in a numpy array with shape = (n_points, dim)
    dim = len(points_str[2].split(' '))
    my_points = np.zeros((n_points, dim), dtype=int)
    points_str = points_str[1:]
    for i in range(n_points):
        point_i = points_str[i].split(' ')
        for j in range(dim):
            # Change position of x and y.
            my_points[i, 1-j] = float(point_i[j])

    return my_points


first_img = cv2.imread("./res01.jpg")
second_img = cv2.imread("./res02.jpg")
# points_1 = get_user_points(first_img, "First Image")
# print("-------------------------------------------")
# points_2 = get_user_points(second_img, "Second Image")
#
# print(points_file_reader("pts1.txt"))
# print(points_file_reader("pts2.txt"))

first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2RGB)
# Read the points from text files:
pts1 = points_file_reader("pts1.txt")
pts2 = points_file_reader("pts2.txt")
# Create a triangulation using Delaunay method:
triangles = Delaunay(pts1)
# Find vertices of each triangle:
vertices1 = pts1[triangles.simplices].astype(np.float32)
vertices2 = pts2[triangles.simplices].astype(np.float32)
# temp = np.zeros(first_img[:,:,0].shape,np.uint8)
# cv2.fillConvexPoly(temp, vertices1[0], 255)
# plt.imshow(temp)
# plt.show()

# Set the number of images:
n_pics = 45
# Shape of the image:
h = first_img.shape[0]
w = first_img.shape[1]

for i in range(0, n_pics):
    # For each time step:
    # Calculate the weighted average of the vertices as explained in the class:
    intermediate_vertices = (1 - i / (n_pics - 1)) * vertices1 + i / (n_pics - 1) * vertices2
    intermediate_vertices = intermediate_vertices.astype(np.int32)
    # print(intermediate_vertices.shape)
    # print(intermediate_vertices)
    # break
    out_temp = np.zeros(first_img.shape, np.uint8)
    for j in range(intermediate_vertices.shape[0]):
        # For each triangle:
        # Find an affine mapping from the corresponding triangles in the first and second image
        trans_mat1 = cv2.getAffineTransform(vertices1[j], intermediate_vertices[j].astype(np.float32))
        trans_mat2 = cv2.getAffineTransform(vertices2[j], intermediate_vertices[j].astype(np.float32))
        # Warp the images using the above mappings
        warped1 = cv2.warpAffine(first_img, trans_mat1, (w, h))
        warped2 = cv2.warpAffine(second_img, trans_mat2, (w, h))
        # Create a mask to change only the values of the j-th triangle
        j_th_triangle = np.zeros(first_img.shape, np.uint8)
        cv2.fillConvexPoly(j_th_triangle, intermediate_vertices[j], (1, 1, 1))
        j_th_triangle = j_th_triangle.astype(bool)
        # Again create a weighted average of two images as explained in the class
        temp = ((1 - i / (n_pics - 1)) * warped1 + i / (n_pics - 1) * warped2).astype(np.uint8)
        # Change the pixels in the j-th triangle
        out_temp[j_th_triangle] = temp[j_th_triangle]
    # Save intermediate images
    plt.imsave("./morphing/morph" + "{:03d}".format(i + 1) + ".jpg", out_temp.astype(np.uint8))
    plt.imsave("./morphing/morph" + "{:03d}".format(2 * n_pics - i) + ".jpg", out_temp.astype(np.uint8))
    # Save step 15 and 30
    if i == 14:
        plt.imsave("res03.jpg", out_temp.astype(np.uint8))
    if i == 29:
        plt.imsave("res04.jpg", out_temp.astype(np.uint8))
