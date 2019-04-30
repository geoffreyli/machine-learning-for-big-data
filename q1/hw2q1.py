import numpy as np
import matplotlib.pyplot as plt

faces = np.loadtxt('./q1/data/faces.csv', delimiter=',')


# Part (b): PCA

# Part (b.1)

sigma = 1/len(faces)*(faces.T@faces)

sigma_eigval, sigma_eigvec = np.linalg.eigh(sigma)

# np.save('./q1/sigma_eigval', sigma_eigval)
# np.save('./q1/sigma_eigvec', sigma_eigvec)
#
# sigma_eigval = np.load('./q1/sigma_eigval.npy')
# sigma_eigvec = np.load('./q1/sigma_eigvec.npy')

# Sort from largest to smallest eigenvalues
sort_idx = sigma_eigval.argsort()[::-1]
sigma_eigval = sigma_eigval[sort_idx]
sigma_eigvec = sigma_eigvec[:, sort_idx]

for i in [1, 2, 10, 30, 50]:
    print("Eigenvalue", i, sigma_eigval[i-1])

# Eigenvalue 1 781.8126992600018
# Eigenvalue 2 161.1515749673267
# Eigenvalue 10 3.3395867548877773
# Eigenvalue 30 0.8090877903777225
# Eigenvalue 50 0.38957773951814506

sigma_trace = np.trace(sigma)
print(sigma_trace)
# 1084.2074349947673

# Part (b.2)

plt.figure(0)
plt.plot(range(1, 51), list(map(lambda k: 1 - np.sum(sigma_eigval[0:k])/sigma_trace, range(1, 51))))
plt.title('Fractional Reconstruction Error vs. k (up to 50)')
plt.xlabel('k')
plt.ylabel('Fractional Reconstruction Error')
plt.savefig('./q1/frac-recon-error.png')
plt.close(0)


# Part (b.3)

# The first eigenvalue captures the line fit to least sum of squares. It's much larger
# than the other eigenvalues because it accounts for the most variation in the data.
# It corresponds to variation in lighting - which is the component that most separates
# the images.


# Part (c): Visualization of the Eigen-Directions

# Part (c.1)

# Plot first 10 raw faces
# plt.figure(1)
# fig, ax = plt.subplots(1, 10, figsize=(20, 5),
#                        subplot_kw={'xticks':[], 'yticks':[]})
# start = 0
# for i in range(start, start+10):
#     ax[i-start].imshow(faces[i, :].reshape(84, 96).T, cmap='Greys_r')
# plt.savefig('./q1/first-10-faces.png')
# plt.close(1)


# Plot first 10 eigenfaces
plt.figure(2)
fig, ax = plt.subplots(1, 10, figsize=(20, 5),
                       subplot_kw={'xticks':[], 'yticks':[]})
start = 0
for i in range(start, start+10):
    ax[i-start].imshow(sigma_eigvec[:, i].reshape(84, 96).T, cmap='Greys_r')
plt.savefig('./q1/first-10-eigenfaces.png')
plt.close(2)


# Part (d.1): Visualization and Reconstruction

imgs = [0, 23, 64, 67, 256]  # row index of original images
k = [1, 2, 5, 10, 50]  # for Top k eigenvectors

# Plot original image sample
plt.figure(3)
fig, ax = plt.subplots(1, 5, figsize=(20, 5),
                       subplot_kw={'xticks':[], 'yticks':[]})
for i in range(0, len(imgs)):
    ax[i].imshow(faces[imgs[i], :].reshape(84, 96).T, cmap='Greys_r', vmin=0, vmax=1)
plt.savefig('./q1/d1-5-orig-faces.png')
plt.close(3)

# Reconstruct original faces from top k eigenvectors
plt.figure('./q1/recon-face-all.png')
fig, ax = plt.subplots(5, 5, figsize=(10, 10),
                       subplot_kw={'xticks': [], 'yticks': []})
for i in range(len(imgs)):
    for j in range(len(k)):
        ax[i, j].imshow((faces[imgs[i], :] @ sigma_eigvec[:, 0:k[j]] @ sigma_eigvec[:, 0:k[j]].T).reshape(84, 96).T,
                   cmap='Greys_r', vmin=0, vmax=1)
plt.savefig('./q1/recon-face-all.png')
plt.close('./q1/recon-face-all.png')












