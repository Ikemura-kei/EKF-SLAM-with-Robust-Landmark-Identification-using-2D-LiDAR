from scipy.odr import RealData, ODR, Model
import numpy as np
import time
import matplotlib.pyplot as plt

# y = m1 * x + c1 = m2 * x + c2
# (m1 - m2) * x = c2 - c1
# x = (c2 - c1) / (m1 - m2)
# y = m1 * x + c1

def line_func(params, x):
    """generate y values given x values and line parameters

    Args:
        params (list of 2 floats): m and c parameter of the line repectively
        x (np.ndarray): a float array of x values, in shape (N, )

    Returns:
        np.ndarray: a float array of computed y values, in shape (N, )
    """
    return params[0] * x + params[1]

def fit_line(pnts):
    """fit a line (compute m and c parameter)

    Args:
        pnts (np.ndarray): an array of points, in shape (N, 2), where the second axis corresponds to x and y values respectively

    Returns:
        tuple of 2 floats: m and c parameters respectively
    """
    data = RealData(pnts[:,0], pnts[:,1])
    model = Model(line_func)
    odr = ODR(data, model, beta0=[0, 0])
    result = odr.run()
    return result.beta # (m, c)

def get_line_scan(bearings):
    """get line parameters of scan lines given bearings of the scan lines

    Args:
        bearings (np.ndarray): 1D array of bearings, in shape (N, )

    Returns:
        tuple of 2 np.ndarray: the first is all of the m parameters, and the second is all c parameters
    """
    return (np.sin(bearings) / np.cos(bearings)), np.zeros_like(bearings)

def get_distance_p2p(pnts1, pnts2):
    """get point-to-point distances

    Args:
        pnts1 (np.ndarray): points arranged in shape (N, 2)
        pnts2 (np.ndarray): points arranged in shape (N, 2)

    Returns:
        np.ndarray: distances of shape (N, )
    """
    diff = pnts1 - pnts2
    dist = np.sqrt(np.sum(diff ** 2, axis=-1)) # (N, )
    return dist

def get_distance_p2l(pnts, m, c):
    """get distance between points and a line

    Args:
        pnts (np.ndarray): points arranged in shape (N, 2)
        m (float): the m parameter of the line
        c (float): the c parameter of the line

    Returns:
        np.ndarray: distances of shape (N, )
    """
    a = m
    b = -1
    return np.abs(a*pnts[...,0] + b*pnts[...,1] + c) / np.sqrt(a**2 + b**2)

def line_intersection(m1, c1, m2s, c2s):
    """find point of intersection between a target line and one or more source lines

    Args:
        m1 (float): the m parameter of the target line
        c1 (float): the c parameter of the target line
        m2s (np.ndarray): an array of m parameters of the source line
        c2s (np.ndarray): an array of c parameters of the source line

    Returns:
        tuple of 2 np.ndarray: x and y points, each is in shape (N, )
    """
    x = (c2s - c1) / (m1 - m2s)
    y = m1 * x + c1
    return x, y

def seed_segment_detection(pnts, ranges, bearings, epsilon, delta, S_NUM, P_MIN, verbose=False):
    NP = pnts.shape[0]
    
    this_segment_valid = True # this flag indicates if the current segment candidate is valid
    
    for i in range(1, NP-P_MIN, 2):
        j = i + S_NUM
        
        # -- fit a candidate segment --
        m, c = fit_line(pnts[i:j])
        
        # -- check the conditions on this candidate segment --
        
        # -- condition 1: predicted point, i.e. intersection of a scan line and the fitted line, should be close to observation for all seeds --
        pred_ms, pred_cs = get_line_scan(bearings=bearings[i:j]) # get scan line parameters from bearings
        predicted_x, predicted_y = line_intersection(m, c, pred_ms, pred_cs) # compute intersection of fitted line and the scan lines 
        
        if verbose:
            for k in range(len(pred_ms)):
                k_ = k + i
                print("--> Predicted and observed points: ({}, {}) | ({}, {})".format(predicted_x[k], predicted_y[k], pnts[k_, 0], pnts[k_, 1]))
        dist_p2p = get_distance_p2p(pnts[i:j], np.stack([predicted_x, predicted_y], axis=-1))
        
        if np.any((dist_p2p > delta)):
            this_segment_valid = False
            
        if verbose:
            print(dist_p2p)
            print("--> Condition 1 passed? ", not np.any((dist_p2p > delta)))
        
        # -- condition 2: all points used to fit the line should be close enough to the line --
        if this_segment_valid:
            dist_p2l = get_distance_p2l(pnts[i:j], m, c)
            this_segment_valid = not np.any((dist_p2l > epsilon))
            
            if verbose:
                print(dist_p2l)
                print("--> Condition 2 passed? ", not np.any((dist_p2l > epsilon)))
        
        if this_segment_valid:
            return [i,j,m,c]
        
    return None

def seed_segment_growing(seed, pnts, P_MIN, L_MIN, epsilon, L_MAX, verbose=False):
    i, j, m, c = seed
    m_prev = m_cur = m
    c_prev = c_cur = c
    
    # -- grow at tail --
    j = j + 1
    while (j < len(pnts)) and get_distance_p2l(pnts[j,...][None,:], m, c)[0] < epsilon:
        if verbose:
            print("--> Distance from a new point at tail to line: ", get_distance_p2l(pnts[j,...][None,:], m, c)[0])
        # # -- refit the line --
        m_prev = m_cur
        c_prev = c_cur
        m_cur, c_cur = fit_line(pnts[i:j])
        j += 1
    if verbose:
        print("--> Distance from a new point at tail to line: ", get_distance_p2l(pnts[j,...][None,:], m, c)[0])
    j = j - 1
    
    # -- grow at head
    i = i - 1
    while (i >= 0) and get_distance_p2l(pnts[i,...][None,:], m, c)[0] < epsilon:
        if verbose:
            print("--> Distance from a new point at head to line: ", get_distance_p2l(pnts[i,...][None,:], m, c)[0])
        # # -- refit the line --
        m_prev = m_cur
        c_prev = c_cur
        m_cur, c_cur = fit_line(pnts[i:j])
        i -= 1
    if verbose:
        print("--> Distance from a new point at head to line: ", get_distance_p2l(pnts[i,...][None,:], m, c)[0])
    i = i + 1
    
    # m_prev, c_prev = fit_line(pnts[i:j])
    line_length = get_distance_p2p(pnts[j,...][None,:], pnts[i,...][None,:])[0]
    if verbose:
        print("--> The length of the grown segment: ", line_length)
    # if (j - i + 1) >= P_MIN and line_length >= L_MIN and line_length < 2.1:
    if (j - i + 1) >= P_MIN and line_length >= L_MIN and line_length < L_MAX:
        return [i, j, m_prev ,c_prev]
    
    return None

def seed_segment_overlap(seed, pnts):
    print("--> Number of segments: {}".format(len(seed)))
    for i in range(len(seed)-1):
        s1, n1, m1, c1 = seed[i]
        s2, n2, m2, c2 = seed[i+1]
        
        if s2 < n1:
            dist_p2l_1 = get_distance_p2l(pnts[s2:n1], m1, c1)
            dist_p2l_2 = get_distance_p2l(pnts[s2:n1], m2, c2)
            
            compare = dist_p2l_1 >= dist_p2l_2
            if not np.any(compare): # if there is at least one occurance where a point in overlap is closer to line 2
                k = np.argmax(compare) # find the index of the first True occurance (caution: when there is no True at all, it will return 0, which is confusing, so we split cases)
            else: # if all points are closer to line 1
                k = len(compare) - 1
            
            n1 = s2 + (k-1)
            s2 = s2 + k

            # print("--> Overlap detected between segment {} and segment {}".format(i, i+1))
        else:
            # print("--> No overlap detected between segment {} and segment {}".format(i, i+1))
            continue
        
        # -- refit lines --
        m1, c1 = fit_line(pnts[s1:n1])
        m2, c2 = fit_line(pnts[s2:n2])
        seed[i] = [s1, n1, m1, c1]
        seed[i+1] = [s2, n2, m2, c2]
        
    return seed

def end_point_generation(seed, pnts):
    N = len(seed)
    if N == 0:
        return None, None
    seed = np.array(seed)
    
    a = seed[:,2] # m
    b = np.ones((N)) * -1
    c = seed[:,3] # c
    
    b_sqr = b ** 2
    a_sqr = a ** 2
    ab = a * b
    bc = b * c
    ac = a * c
    a_sqr_plus_b_sqr = a_sqr + b_sqr
    
    s = seed[:,0].astype(np.int64)
    e = seed[:,1].astype(np.int64)

    s_pnts = pnts[s]
    e_pnts = pnts[e]
    
    new_s_pnts = np.zeros_like(s_pnts)
    new_e_pnts = np.zeros_like(e_pnts)
    
    new_s_pnts[:,0] = (b_sqr * s_pnts[:,0] - ab * s_pnts[:,1] - ac) / a_sqr_plus_b_sqr
    new_s_pnts[:,1] = (a_sqr * s_pnts[:,1] - ab * s_pnts[:,0] - bc) / a_sqr_plus_b_sqr
    
    new_e_pnts[:,0] = (b_sqr * e_pnts[:,0] - ab * e_pnts[:,1] - ac) / a_sqr_plus_b_sqr
    new_e_pnts[:,1] = (a_sqr * e_pnts[:,1] - ab * e_pnts[:,0] - bc) / a_sqr_plus_b_sqr
    
    return new_s_pnts, new_e_pnts

def landmark_extraction(pnts, ranges, bearings, epsilon, delta, S_NUM, P_MIN, L_MIN, L_MAX, verbose=False):
    seed_segments = []
    i = 0
    END = pnts.shape[0] - 1
    
    while i <= END:
        if verbose:
            print("--> Current breakpoint: ", i)
            
        seed = seed_segment_detection(pnts[i:], ranges[i:], bearings[i:], epsilon, delta, S_NUM, P_MIN, verbose=verbose)
        if seed is None:
            i += 1
            continue
        
        i1,j1,m1,c1 = seed
        i1 = i1 + i
        j1 = j1 + i
        
        grown_seed = seed_segment_growing([i1,j1,m1,c1], pnts, P_MIN, L_MIN, epsilon, L_MAX, verbose=verbose)
        if grown_seed is None:
            i2 = i1
            j2 = j1
            m2 = m1
            c2 = c1
        else: 
            i2,j2,m2,c2 = grown_seed
            seed_segments.append([i2,j2,m2,c2])
            
        i = j2
        
        # seed_segments.append([i2,j2,m2,c2])
        
    # -- deal with overlaps --
    # seed_segments = seed_segment_overlap(seed_segments, pnts)
        
    # -- generate endpoints --
    s, e = end_point_generation(seed_segments, pnts)
    
    return s, e

class GaoEtAl():
    def __init__(self, EPSILON=0.03, DELTA=0.10124, S_NUM=4, P_MIN=8, L_MIN=0.345, L_MAX=2.1):
    # def __init__(self, EPSILON=0.102, DELTA=0.15, S_NUM=4, P_MIN=7, L_MIN=0.5, L_MAX=3):
        self.EPSILON = EPSILON
        self.DELTA = DELTA
        self.S_NUM = S_NUM
        self.P_MIN = P_MIN
        self.L_MIN = L_MIN
        self.L_MAX = L_MAX
        
    def compute(self, pnts, ranges, bearings):
        # -- subsampling to reduce computational time --
        pnts = pnts[::2]
        ranges = ranges[::2]
        bearings = bearings[::2]
        return landmark_extraction(pnts, ranges, bearings, self.EPSILON, self.DELTA, self.S_NUM, self.P_MIN, self.L_MIN, self.L_MAX)

def visualize_segments(pnts, s, e):
    if s is None or e is None or pnts is None:
        return

    plt.scatter(pnts[:,0], pnts[:,1], s=2.5)
    plt.scatter(0, 0, s=20.5)
    plt.xlim([-2, 7])
    plt.ylim([-5, 4])
    
    for i in range(len(s)):
        plt.plot(np.array([s[i,0], e[i,0]]), np.array([s[i,1], e[i,1]]), 'r', linestyle="--")
        
    plt.show()

if __name__ == "__main__":
    # -- define parameters --
    EPSILON = 0.0575
    DELTA = 0.1
    S_NUM = 4
    P_MIN = 6
    L_MIN = 0.35 # m

    # -- generate simulated laser points --
    data = np.load("./playground/laser_data.npy") # (x, y, ranges, bearings)
    laser_points = data[:,:2] # retrieve x and y
    ranges = data[:,2]
    bearings = data[:,-1]

    # -- add a bit of noise --
    N, _ = laser_points.shape
    laser_points = laser_points + np.random.normal(0, 0.0095, (N, 2))

    # -- run the landmark extraction algorithm --
    start_t = time.time()
    s, e = landmark_extraction(laser_points, ranges, bearings, EPSILON, DELTA, S_NUM, P_MIN, L_MIN, False)
    end_t = time.time()
    exe_time = end_t - start_t
    print("--> Computational time {}".format(exe_time))

    # -- visualization --
    fig = plt.figure()
    plt.scatter(laser_points[:,0], laser_points[:,1], s=2.5)
    plt.scatter(0, 0, s=20.5)
    pred_m, pred_c = get_line_scan(bearings[0])
    plt.axline(xy1=(0,pred_c), slope=pred_m, c='g')
    plt.xlim([-2, 7])
    plt.ylim([-5, 4])
    
    for i in range(len(s)):
        plt.plot(np.array([s[i,0], e[i,0]]), np.array([s[i,1], e[i,1]]), 'r', linestyle="--")
        
    plt.show()
        