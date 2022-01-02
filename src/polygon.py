import numpy as np

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def find_intersection(p1,  p2,  p3,  p4):
    """Find the point of intersection between two line.

    The two lines are p1 --> p2 and p3 --> p4.
    Reference:http://csharphelper.com/blog/2020/12/enlarge-a-polygon-that-has-colinear-vertices-in-c/

    :param p1: line 1's start point, 2D point (x,y)
    :type p1: list
    :param p2: line 1's end point
    :type p2: list
    :param p3: line 2's start point
    :type p3: list
    :param p4: line 2's end point
    :type p4: list
    :return: The intersection point of two line
    :rtype: list
    """
    # the segments
    dx12 = p2[0] - p1[0]
    dy12 = p2[1] - p1[1]
    dx34 = p4[0] - p3[0]
    dy34 = p4[1] - p3[1]

    denominator = (dy12 * dx34 - dx12 * dy34)
    if denominator == 0:
        # The two lines are parallel
        return None

    t1 = ((p1[0] - p3[0]) * dy34 + (p3[1] - p1[1]) * dx34) / denominator

    # point of intersection.
    intersection = [p1[0] + dx12 * t1, p1[1] + dy12 * t1]
    return intersection


def is_clockwise(point_list):
    """Check whether the list is clockwise. 

    :param point_list: the point lists
    :type point_list: numpy 
    :return: yes if the points are clockwise, otherwise is no
    :rtype: bool
    """
    sum = 0
    for i in range(len(point_list)):
        cur = point_list[i]
        next = point_list[(i+1) % len(point_list)]
        sum += (next[0] - cur[0]) * (next[1] + cur[1])
    return sum > 0


def inside_polygon_2d(points_list, polygon_points, on_line=False, eps=1e-4):
    """ Test the points inside the polygon. 
    Implement 2D PIP (Point Inside a Polygon).
    
    :param points_list: The points locations numpy array whose size is [point_numb, 2]. The point storage list is as [[x_1, y_1], [x_2, y_2],...[x_n, y_n]].
    :type points_list: numpy
    :param polygon_points: The clock-wise points sequence. The storage is the same as points_list. size is [triangle_points_tangent, 2]
    :type polygon_points: numpy
    :param on_line: The inside point including the boundary points, if True. defaults to False
    :type on_line: bool, optional
    :param eps: Use the set the polygon's line width. The distance between two pixel. defaults to 1e-4
    :type eps: float, optional
    :return: A numpy Boolean array, True is inside the polygon, False is outside.
    :rtype: numpy
    """
    point_inside = np.full(np.shape(points_list)[0], False, dtype=bool)  # the point in the polygon
    online_index = np.full(np.shape(points_list)[0], False, dtype=bool)  # the point on the polygon lines

    points_x = points_list[:, 0]
    points_y = points_list[:, 1]

    def GREATER(a, b): return a >= b
    def LESS(a, b): return a <= b

    # try each line segment
    for index in range(np.shape(polygon_points)[0]):
        polygon_1_x = polygon_points[index][0]
        polygon_1_y = polygon_points[index][1]

        polygon_2_x = polygon_points[(index + 1) % len(polygon_points)][0]
        polygon_2_y = polygon_points[(index + 1) % len(polygon_points)][1]

        # exist points on the available XY range
        test_result = np.logical_and(GREATER(points_y, min(polygon_1_y, polygon_2_y)), LESS(points_y, max(polygon_1_y, polygon_2_y)))
        test_result = np.logical_and(test_result, LESS(points_x, max(polygon_1_x, polygon_2_x)))
        if not test_result.any():
            continue

        # get the intersection points
        if LESS(abs(polygon_1_y - polygon_2_y), eps):
            test_result = np.logical_and(test_result, GREATER(points_x, min(polygon_1_x, polygon_2_x)))
            intersect_points_x = points_x[test_result]
        else:
            intersect_points_x = (points_y[test_result] - polygon_1_y) * \
                (polygon_2_x - polygon_1_x) / (polygon_2_y - polygon_1_y) + polygon_1_x

        # the points on the line
        on_line_list = LESS(abs(points_x[test_result] - intersect_points_x), eps)
        if on_line_list.any():
            online_index[test_result] = np.logical_or(online_index[test_result], on_line_list)

        # the point on the left of the line
        if LESS(points_x[test_result], intersect_points_x).any():
            test_result[test_result] = np.logical_and(test_result[test_result], LESS(points_x[test_result], intersect_points_x))
            point_inside[test_result] = np.logical_not(point_inside[test_result])

    if on_line:
        return np.logical_or(point_inside, online_index).reshape(np.shape(points_list[:, 0]))
    else:
        return np.logical_and(point_inside, np.logical_not(online_index)).reshape(np.shape(points_list[:, 0]))


def enlarge_polygon(old_points, offset):
    """Return points representing an enlarged polygon.

    Reference: http://csharphelper.com/blog/2016/01/enlarge-a-polygon-in-c/
    
    :param old_points: the polygon vertexes, and the points should be in clock wise
    :type: list [[x_1,y_1], [x_2, y_2]......]
    :param offset: the ratio of the polygon enlarged
    :type: float
    :return: the offset points
    :rtype: list
    """
    enlarged_points = []
    num_points = len(old_points)
    for j in range(num_points):
        # 0) find "out" side
        if not is_clockwise(old_points):
            log.error("the points list is not clockwise.")

        # the points before and after j.
        i = (j - 1)
        if i < 0:
            i += num_points
        k = (j + 1) % num_points

        # 1) Move the points by the offset.
        # the points of line parallel to ij
        v1 = np.array([old_points[j][0] - old_points[i][0], old_points[j][1] - old_points[i][1]], np.float)
        norm = np.linalg.norm(v1)
        v1 = v1 / norm * offset
        n1 = [-v1[1], v1[0]]
        pij1 = [old_points[i][0] + n1[0], old_points[i][1] + n1[1]]
        pij2 = [old_points[j][0] + n1[0], old_points[j][1] + n1[1]]

        # the points of line parallel to jk
        v2 = np.array([old_points[k][0] - old_points[j][0], old_points[k][1] - old_points[j][1]], np.float)
        norm = np.linalg.norm(v2)
        v2 = v2 / norm * offset
        n2 = [-v2[1], v2[0]]
        pjk1 = [old_points[j][0] + n2[0], old_points[j][1] + n2[1]]
        pjk2 = [old_points[k][0] + n2[0], old_points[k][1] + n2[1]]

        # 2) get the shifted lines ij and jk intersect
        lines_intersect = find_intersection(pij1, pij2, pjk1, pjk2)
        enlarged_points.append(lines_intersect)

    return enlarged_points
