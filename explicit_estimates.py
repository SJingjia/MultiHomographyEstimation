from uuid import NAMESPACE_X500
import numpy as np
from transform import get_data
import cv2
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
import pickle


def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()

def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent

# calculate the third-order determinant
def mat3det(A):
    return A[0,0]*(A[1,1]*A[2,2]-A[2,1]*A[1,2]) - \
           A[0,1]*(A[1,0]*A[2,2]-A[2,0]*A[1,2]) + \
           A[0,2]*(A[1,0]*A[2,1]-A[2,0]*A[1,1])
        
# calculate eigenvalue
def cal_eigenvalue(A, B, c3):
    c0 = mat3det(A)
    c1 = mat3det(np.c_[(B[:,0], A[:,1], A[:,2])]) + \
         mat3det(np.c_[(A[:,0], B[:,1], A[:,2])]) + \
         mat3det(np.c_[(A[:,0], A[:,1], B[:,2])])
    c2 = mat3det(np.c_[(A[:,0], B[:,1], B[:,2])]) + \
         mat3det(np.c_[(B[:,0], A[:,1], B[:,2])]) + \
         mat3det(np.c_[(B[:,0], B[:,1], A[:,2])])
    
    w = (c1*c2 - 9*c0*c3)/(2*(c2**2-3*c1*c3))
    
    return w

# calculate J 
def cal_rank1J(Hs):
    rank1J = []
    # c3 = np.linalg.det(Hs[0])
    c3 = mat3det(Hs[0])
    for i in range(1, Hs.shape[0]):
        w = cal_eigenvalue(Hs[i], Hs[0], c3)
        rank1J.append(Hs[i] - w*Hs[0])
    rank1J = np.c_[tuple(rank1J)]
    return rank1J

# Get the column indexs list of all 2x2 minors of J
def cal_subindexs(N):
    subindexs = []
    for i in range(N-1):
        for j in range(i, N):
            subindexs.append((i,j))
    return subindexs

# Convert the array of independent variables of size (n,) 
# to pixs0 of size (p,2) and Hs of size (h,3,3)
def to_pixs0_andHs(x, num_pixs):
        N = int(num_pixs*2)
        pixs0_ = x[:N].reshape(num_pixs, 2)
        Hs = x[N:].reshape(-1, 3, 3)
        return pixs0_, Hs                

# Maximum Likelihood Estimation
def ML_estimation(init_x, pixs0, pixs1, pixs_label):
    num_pixs = len(pixs0)
    # def constraint_func(x):
    #     add H_norm = 1
    #     pixs0_, Hs = to_pixs0_andHs(x, num_pixs)
    #     rank1J = cal_rank1J(Hs)
    #     numH = Hs.shape[0]
    #     const = 0
    #     # Hnorms = [np.linalg.norm(H) for H in Hs]
    #     Hnorms = [np.sum(H**2) for H in Hs]
    #     for a, b in [(0,1), (0,2), (1,2)]:
    #         for c, d in cal_subindexs(numH):
    #             const += (rank1J[a,c]*rank1J[b,d] - rank1J[a,d]*rank1J[b,c])**2/(Hnorms[int(c/3)]*Hnorms[int(d/3)])**2
    #     return const
    
    # explicit constraints
    def constraint_func(x):
        pixs0_, Hs = to_pixs0_andHs(x, num_pixs)
        rank1J = cal_rank1J(Hs)
        numH = Hs.shape[0]
        const = 0
        # Hnorms = [np.linalg.norm(H) for H in Hs]
        # Hnorms = [np.sum(H**2) for H in Hs]
        for a, b in [(0,1), (0,2), (1,2)]:
            for c, d in cal_subindexs(numH):
                const += (rank1J[a,c]*rank1J[b,d] - rank1J[a,d]*rank1J[b,c])**2
        
        return const
    
    # JML
    def cost_func(x, pixs0, pixs1, pixs_label):
        # num_pixs = pixs0.shape[0]
        pixs0_, Hs = to_pixs0_andHs(x, num_pixs)
        JML = 0
        H_index = 0
        for i in np.unique(pixs_label):
            indexs = np.where(pixs_label == i)
            oneh_pixs0_ = pixs0_[indexs]
            oneh_pixs0 = pixs0[indexs]
            oneh_pixs1 = pixs1[indexs]
            trans_oneh_pixs0_ = cv2.perspectiveTransform(oneh_pixs0_.reshape(-1, 1, 2), Hs[H_index]).reshape(-1,2)
            JML += np.sum((oneh_pixs0_ - oneh_pixs0) ** 2) + np.sum((oneh_pixs1 - trans_oneh_pixs0_)**2)
            H_index += 1
        return JML
    
    nlc = NonlinearConstraint(constraint_func, -np.inf, 0)
    
    # ? with constraints norm(H) = 1
    # nlc = [NonlinearConstraint(constraint_func, -np.inf, 0)]
    # pixs0_, Hs = to_pixs0_andHs(x, num_pixs)
    # numH = Hs.shape[0]
    
    # for i in range(numH):
    #     def Hconstraint(x):
    #         pixs0_, Hs = to_pixs0_andHs(x, num_pixs)
    #         return np.sqrt(np.sum(Hs[i]**2))
    #     nlc.append(NonlinearConstraint(Hconstraint, 1, 1))
    
    # 'trust-constr' with interior point algorithm
    res = minimize(cost_func, init_x, args = (pixs0, pixs1, pixs_label), method = 'trust-constr', constraints = nlc)
    
    # ? 'COBYLA' method
    # cons = ({'type': 'ineq', 'fun': constraint_func})
    # res = minimize(cost_func, init_x, args = (pixs0, pixs1, pixs_label), method = 'COBYLA', constraints = cons)
    
    return res
    
      
            
          
        

if __name__ == '__main__':

    # downsampling = 500
    # dataset = get_data([0, 1], downsampling=downsampling)
    # np.random.seed(0)
    # for data in dataset:
    for i in range(1):
        # img0, img1, mpixs0_downsample, mpixs1_downsample, pixs_label, name0, name1 = data
        img0, img1, mpixs0_downsample, mpixs1_downsample, pixs_label = loadList(f"./data/data_{i}.pkl")
        # saveList([img0, img1, mpixs0_downsample, mpixs1_downsample, pixs_label], f"./data/data_{i}.pkl")
        mpixs0_downsample_noise = mpixs0_downsample + np.random.normal(0, 1, mpixs0_downsample.shape)
        Hs = []
        init_reproject_errors = []
        init_noise_reproject_errors = []
        init_pixs0_errors = []
        for i in np.unique(pixs_label):
            indexs = np.where(pixs_label == i)
            print("num of oneH pts: ", indexs[0].shape[0])
            oneh_pixs0 = mpixs0_downsample[indexs]
            # oneh_pixs0_noise = oneh_pixs0 + np.random.normal(0, 1, oneh_pixs0.shape)
            oneh_pixs0_noise = mpixs0_downsample_noise[indexs]
            oneh_pixs1 = mpixs1_downsample[indexs]
            H, mask = cv2.findHomography(oneh_pixs0_noise, oneh_pixs1, method = cv2.RANSAC)
            if type(H) == np.ndarray:
                H /= np.linalg.norm(H)
                Hs.append(H)
                print(H)
                oneh_pixs0_p = cv2.perspectiveTransform(oneh_pixs0.reshape(-1, 1, 2), H)
                oneh_pixs0_noise_p = cv2.perspectiveTransform(oneh_pixs0_noise.reshape(-1, 1, 2), H)
                # print(np.mean(oneh_pixs0_p.reshape(-1, 2) - oneh_pixs1))
                init_reproject_errors.append(np.mean(oneh_pixs0_p.reshape(-1, 2) - oneh_pixs1))
                init_noise_reproject_errors.append(np.mean(oneh_pixs0_noise_p.reshape(-1, 2) - oneh_pixs1))
                init_pixs0_errors.append(np.mean(oneh_pixs0 - oneh_pixs0_noise))
            else:
                print(H)
                mpixs0_downsample = np.delete(mpixs0_downsample, indexs, axis = 0)
                mpixs0_downsample_noise = np.delete(mpixs0_downsample_noise, indexs, axis = 0)
                mpixs1_downsample = np.delete(mpixs1_downsample, indexs, axis = 0)
                pixs_label = np.delete(pixs_label, indexs, axis = 0)
        
        init_pixs0_ = mpixs0_downsample_noise.reshape(-1)
        init_Hs = np.array(Hs).reshape(-1)
        x = np.hstack([init_pixs0_, init_Hs])
        res = ML_estimation(x, mpixs0_downsample_noise, mpixs1_downsample, pixs_label)
        # print(res)
        # print("method: ", res['method'])
        opt_pixs0_, opt_Hs = to_pixs0_andHs(res['x'], mpixs0_downsample_noise.shape[0])
        
        H_index = 0
        opt_reproject_errors = []
        opt_noise_reproject_errors = []
        opt_pixs0_errors = []
        for i in np.unique(pixs_label):
            indexs = np.where(pixs_label == i)
            oneh_pixs0 = mpixs0_downsample[indexs]
            # oneh_pixs0_noise = oneh_pixs0 + np.random.normal(0, 1, oneh_pixs0.shape)
            oneh_pixs0_noise = mpixs0_downsample_noise[indexs]
            opt_oneh_pixs0_noise = opt_pixs0_[indexs]
            oneh_pixs1 = mpixs1_downsample[indexs]
            opt_H = opt_Hs[H_index]
            init_H = Hs[H_index]
            
            print("fro norm of opt_H: ", np.linalg.norm(opt_H))
            print(opt_H, init_H)
            
            oneh_pixs0_p = cv2.perspectiveTransform(oneh_pixs0.reshape(-1, 1, 2), opt_H)
            opt_oneh_pixs0_noise_p = cv2.perspectiveTransform(opt_oneh_pixs0_noise.reshape(-1, 1, 2), opt_H)
            
            # print("H error: ", np.mean(oneh_pixs0_p.reshape(-1, 2) - oneh_pixs1),
            #       "pixs0 error: ", np.mean(oneh_pixs0 - opt_oneh_pixs0_noise))
            opt_reproject_errors.append(np.mean(oneh_pixs0_p.reshape(-1, 2) - oneh_pixs1))
            opt_noise_reproject_errors.append(np.mean(opt_oneh_pixs0_noise_p.reshape(-1, 2) - oneh_pixs1))
            opt_pixs0_errors.append(np.mean(oneh_pixs0 - opt_oneh_pixs0_noise))
            print(np.mean(oneh_pixs0_noise - opt_oneh_pixs0_noise))
            
            H_index += 1    
        
        print("init_reproject_errors: ", np.mean(np.sqrt(np.array(init_reproject_errors)**2)), init_reproject_errors)
        print("opt_reproject_errors: ",np.mean(np.sqrt(np.array(opt_reproject_errors)**2)), opt_reproject_errors)
        print("init_noise_reproject_errors: ", np.mean(np.sqrt(np.array(init_noise_reproject_errors)**2)), init_noise_reproject_errors)
        print("opt_noise_reproject_errors: ", np.mean(np.sqrt(np.array(opt_noise_reproject_errors)**2)), opt_noise_reproject_errors)
        print("init_pixs0_errors: ", np.mean(np.sqrt(np.array(init_pixs0_errors)**2)), init_pixs0_errors)
        print("opt_pixs0_errors: ", np.mean(np.sqrt(np.array(opt_pixs0_errors)**2)), opt_pixs0_errors)
            
        
    
    
        
            
        
