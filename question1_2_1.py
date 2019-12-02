import pandas as pd
import numpy as np
import math
import os
import shutil
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score

# TODO Plotar grafico PCA para resultado
# TODO rodar para centrois randomicos e centroides pelos dados
# TODO Salvar resultados de melhor rand junto com J

# Calculate Kernel
def gaussian(x,v,sigma):
    return math.exp((-((x-v)**2))/(sigma))


# Read data from Image Segmentation Database
data = pd.read_csv('data/seg.test')

# Load ground thruth labels
ground_truth = np.genfromtxt('data/test_gt.csv', delimiter=',')

# Splits into shape view and rgb view
# First 9 features
# shape_view([2100]points (n), [9]features (p))
shape_view = data.values[:, 0:9]
# 10 Remaining features
# rgb_view([2100]points (n), [10]features (p))
rgb_view = data.values[:, 9:19]

# Remove less representative features in columns 2, 3 and 4
shape_view = shape_view[:,[0,1,5,6,7,8]]

# Normalize data
scaler = MinMaxScaler()
rgb_view = scaler.fit_transform(rgb_view)
shape_view = scaler.fit_transform(shape_view)

data = {'shape': shape_view}

# Number of Clusters
c = 7
# Fuzziness of membership
m = 1.6
# Iteration limit
T = 150
# Error threshold
e = 10e-10
# Number of Epochs
ep = 100

if not os.path.isdir("results"):
    os.makedirs("results")

for name, view in data.items():

    if os.path.isdir("results/" + name):
        shutil.rmtree("results/" + name)
    else:
        os.makedirs("results/" + name)

    # Number of points
    n = view.shape[0]
    # Number of features
    p = view.shape[1]

    sigma = []
    for j in range(p):
        dist = pdist(view[:,j].reshape(-1,1)) # Size given by Binominal Coefficient
        mean = np.mean([np.quantile(dist, 0.1), np.quantile(dist, 0.9)])
        sigma.append(mean)

    best_J = float("inf")
    best_rand = 0

    for epoch in range(0, 25):
        print("epoch ", epoch+1)
        
        # Randomly initialize the fuzzy membership degree
        # u( clusters (c), points (n))
        u = np.random.rand(c, n)
        sum = np.sum(u, axis = 0) 
        for i in range(n):
            for j in range(c):
                u[j,i] = u[j,i] / sum[i]

        # Initialize weights of the variables
        # lamb(features (p))
        lamb = np.ones(p)

        # Initialize cluster centroids randomly
        # v(clusters (c), features (p))
        # v = np.random.rand(c, p)

        # Initialize cluster centroids from data
        v = np.copy(view)
        np.random.shuffle(v)
        v = v[0:c, :]

        J = float("inf")
        rand = 0

        for it in range(T):
            # print("iteration ", it+1)

            # Update cluster centrois v
            for i in range(c):
                for j in range(p):
                    a = 0
                    b = 0
                    for k in range(n):
                        a += ((u[i,k])**m) * gaussian(view[k,j], v[i,j], sigma[j]) * view[k,j]
                        b +=  ((u[i,k])**m) * gaussian(view[k,j], v[i,j], sigma[j])
                    v[i,j] = a / b

            # Update features weights
            a = 1
            for j in range(p):
                b = 0
                for i in range(c):
                    for k in range(n):
                        b += ((u[i,k])**m) * (2 * (1 - gaussian(view[k,j], v[i,j], sigma[j])))
                a *= b
            for j in range(p):
                b = 0
                for i in range(c):
                        for k in range(n):
                            b += ((u[i,k])**m) * (2 * (1 - gaussian(view[k,j], v[i,j], sigma[j])))

                lamb[j] = (a ** (1/p)) / b
                
            # Update fuzzy membership degree 
            for i in range(c):
                for k in range(n):
                    a = 0
                    for h in range(c):
                        phi_a = 0
                        phi_b = 0
                        for j in range(p):
                            phi_a += lamb[j] * 2 *(1 - gaussian(view[k,j], v[i,j], sigma[j]))
                            phi_b += lamb[j] * 2 *(1 - gaussian(view[k,j], v[h,j], sigma[j]))
                        a += (phi_a / phi_b)**(1/(m-1))
                    u[i,k] = a ** (-1)

            # Calculate J 
            J_prev = J
            J = 0
            for i in range(c):
                for k in range(n):
                    phi = 0
                    for j in range(p):
                        phi += lamb[j] * 2 *(1 - gaussian(view[k,j], v[i,j], sigma[j]))
                    J +=  ((u[i,k])**m) * phi
            
            # Defuzzyfy
            crisp = np.argmax(u, axis=0)

            # # print ajusted rand index
            # for i in range(7):
            #     print("Number of points in cluster " + str(i+1) + ": " + str(np.count_nonzero(crisp == i)))
            # print(J)
            # print("Adjusted rand index: " + str(adjusted_rand_score(ground_truth, crisp)))
            rand = adjusted_rand_score(ground_truth, crisp)

            # Checks if error is reducing with iterations
            if (J_prev - J) < e:
                if J_prev < J:
                    print("ERROR!")
                break
        # Save best J results
        if J < best_J:
            best_J = J
            i = 1
            while(True):
                res_dir = "results/" + name + "_J/" + name + "_" + str(i)
                if not os.path.isdir(res_dir):
                    os.makedirs(res_dir)
                    np.savetxt(res_dir + "/best_u.csv", u, delimiter=",")
                    np.savetxt(res_dir + "/best_lamb.csv", lamb, delimiter=",")
                    np.savetxt(res_dir + "/best_v.csv", v, delimiter=",")
                    np.savetxt(res_dir + "/best_J.csv", np.array([J]), delimiter=",")
                    break
                i += 1
        print("J: " + str(J))
        print("Best J: " + str(best_J))

        # Save best rand results
        if  rand > best_rand:
            best_rand = rand
            i = 1
            while(True):
                res_dir = "results/" + name + "_rand/" + name + "_" + str(i)
                if not os.path.isdir(res_dir):
                    os.makedirs(res_dir)
                    np.savetxt(res_dir + "/best_u.csv", u, delimiter=",")
                    np.savetxt(res_dir + "/best_lamb.csv", lamb, delimiter=",")
                    np.savetxt(res_dir + "/best_v.csv", v, delimiter=",")
                    np.savetxt(res_dir + "/best_J.csv", np.array([J]), delimiter=",")
                    break
                i += 1
        print("rand: " + str(rand))
        print("Best rand: " + str(best_rand))
