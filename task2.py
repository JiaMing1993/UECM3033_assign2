import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.linalg as sp

#Compress - Lower resolution picture
img=mpimg.imread('pic.jpg')
[r,g,b] = [img[:,:,i] for i in range(3)]

Ur,Sr,Vr=sp.svd(r,full_matrices=False)
Sr = np.diag(Sr)

Ug,Sg,Vg=sp.svd(g,full_matrices=False)
Sg = np.diag(Sg)

Ub,Sb,Vb=sp.svd(b,full_matrices=False)
Sb = np.diag(Sb)

SrNew = np.zeros_like(Sr)
SrNew[0:30] = Sr[0:30]
Rnew = Ur.dot(SrNew).dot(Vr)

SgNew = np.zeros_like(Sg)
SgNew[0:30] = Sg[0:30]
Gnew = Ug.dot(SgNew).dot(Vg)

SbNew = np.zeros_like(Sb)
SbNew[0:30] = Sb[0:30]
Bnew = Ub.dot(SbNew).dot(Vb)

img[:,:,0]=Rnew
img[:,:,1]=Gnew
img[:,:,2]=Bnew

fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img)
ax2.imshow(r, cmap = 'Reds')
ax3.imshow(g, cmap = 'Greens')
ax4.imshow(b, cmap = 'Blues')
plt.show()

#Compress - better resolution picture
img_2=mpimg.imread('pic.jpg')
[r2,g2,b2] = [img_2[:,:,i] for i in range(3)]

Ur2,Sr2,Vr2=sp.svd(r2,full_matrices=False)
Sr2 = np.diag(Sr2)

Ug2,Sg2,Vg2=sp.svd(g2,full_matrices=False)
Sg2 = np.diag(Sg2)

Ub2,Sb2,Vb2=sp.svd(b2,full_matrices=False)
Sb2 = np.diag(Sb2)

SrNew2 = np.zeros_like(Sr2)
SrNew2[0:200] = Sr2[0:200]
Rnew2 = Ur2.dot(SrNew2).dot(Vr2)

SgNew2 = np.zeros_like(Sg2)
SgNew2[0:200] = Sg2[0:200]
Gnew2 = Ug2.dot(SgNew2).dot(Vg2)

SbNew2 = np.zeros_like(Sb2)
SbNew2[0:200] = Sb2[0:200]
Bnew2 = Ub2.dot(SbNew2).dot(Vb2)

img_2[:,:,0]=Rnew2
img_2[:,:,1]=Gnew2
img_2[:,:,2]=Bnew2

fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img_2)
ax2.imshow(r2, cmap = 'Reds')
ax3.imshow(g2, cmap = 'Greens')
ax4.imshow(b2, cmap = 'Blues')
plt.show()