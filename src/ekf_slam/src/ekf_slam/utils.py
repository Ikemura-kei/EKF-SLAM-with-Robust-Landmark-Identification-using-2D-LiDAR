import numpy as np
import matplotlib.pyplot as plt

def pnt_on_line_closest_to(pnts, ms, cs):
    N = pnts.shape[0]
    a = ms
    b = np.ones((N)) * -1
    c = cs
    
    b_sqr = b ** 2
    a_sqr = a ** 2
    ab = a * b
    bc = b * c
    ac = a * c
    a_sqr_plus_b_sqr = a_sqr + b_sqr
    
    new_pnts = np.zeros_like(pnts).astype(np.float32)
    
    new_pnts[:,0] = (b_sqr * pnts[:,0] - ab * pnts[:,1] - ac) / a_sqr_plus_b_sqr
    new_pnts[:,1] = (a_sqr * pnts[:,1] - ab * pnts[:,0] - bc) / a_sqr_plus_b_sqr
    
    return new_pnts

def yaw_to_rot_mat(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def ego2global(agent_pos, pnts):
    # (N, 2)
    homo = yaw_to_rot_mat(agent_pos[2])
    homo[0, 2] = agent_pos[0]
    homo[1, 2] = agent_pos[1]
    
    N = len(pnts)
    pnts_homo = np.concatenate([pnts, np.ones((N, 1))], axis=-1) # (N, 3)
    pnts_homo = pnts_homo[...,None] # (N, 3, 1)
    new_pnts = np.squeeze(np.matmul(homo, pnts_homo), axis=-1) # (N, 3)
    
    # print(homo)
    # print(pnts[0])
    # print(new_pnts[0])
    
    return new_pnts[:,:2] # (N, 2)

if __name__ == "__main__":
    # pose = np.array([2, 0, np.pi])
    # pnts = np.array([[1, 0]])
    # print(ego2global(pose, pnts))
    
    ms = np.array([1])
    cs = np.array([0])
    pnt = np.array([[-2.5, 0.5]])

    plt.axline(xy1=(0, cs[0]), slope=ms[0])
    plt.scatter(x=pnt[:,0], y=pnt[:,1], c='g')
    
    landmark_pos = pnt_on_line_closest_to(pnt, ms, cs)
    plt.scatter(x=landmark_pos[:,0], y=landmark_pos[:,1], c='r')
    
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.show()