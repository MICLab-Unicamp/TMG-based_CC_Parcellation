import numpy as np

# Creates Torus
def sintTorus(l1, l2, l3, n, R, r, l1_back, l2_back, l3_back, background=True):
    
    centro = (n-1.)/2

    x,y,z = np.mgrid[0:n,0:n,0:n] 
    t = np.zeros(x.shape)
    t = (((R-(((x-centro)**2)+((y-centro)**2))**0.5)**2+(z-centro)**2-r**2)<0)

    # Eigenvalues, different for the Torus and the background
    eigvalsT = np.ones(np.append(t.shape,3))
    eigvalsT[...,0] *= l1
    eigvalsT[...,1] *= l2
    eigvalsT[...,2] *= l3
    eigvals_back = np.ones(eigvalsT.shape)
    eigvals_back[...,0] *= l1_back
    eigvals_back[...,1] *= l2_back
    eigvals_back[...,2] *= l3_back
    eigvals = eigvalsT*t[..., None] + (1-t[..., None])*eigvals_back

    # Initialize the background eigenvectors
    eigvectsback = np.zeros(np.append(t.shape,(3,3)))
    if background:
        eigvectsback[...,2,0] = np.ones(t.shape)
        eigvectsback[...,1,1] = np.ones(t.shape)
        eigvectsback[...,0,2] = np.ones(t.shape)

    # Initialize the eigenvectors of the Torus
    eigvectstorus = np.zeros(np.append(t.shape,(3,3)))
    eigvectstorus[...,2,2] = -np.ones(t.shape)

    # Modifies the eigenvectors belonging to the Torus, making them tangent to the surface of the Torus
    temp1 = (centro-y)/((centro-x)**2+(centro-y)**2)**0.5
    temp2 = (x-centro)/((centro-x)**2+(centro-y)**2)**0.5

    temp1[np.nonzero(np.isnan(temp1))] = 0
    temp2[np.nonzero(np.isnan(temp2))] = 0

    eigvectstorus[...,0,0] = temp1
    eigvectstorus[...,1,0] = temp2

    eigvectstorus[...,0,1] = temp2
    eigvectstorus[...,1,1] = -temp1

    # Eigenvectors for the entire image (combining background and torus)
    eigvects = eigvectsback*(1-t[..., None, None])+eigvectstorus*t[..., None, None]

    return eigvals,eigvects

# Creates Kissing
def sintKissing(l1, l2, l3, n, R, r, l1_back, l2_back, l3_back, background=True):
    
    # Creates Torus
    eigvals, eigvects = sintTorus(l1, l2, l3, n, R, r, l1_back, l2_back, l3_back, background)
    
    # Manipulate the Torus to generate Kissing
    nd, sx, sy, sz = eigvals.shape
    
    aux = eigvects[int(sx/2):,:,:,:,:]
    eigvects[:int(sx/2),:,:,:,:] = aux
    eigvects[int(sx/2):,:,:,:,:] = aux[::-1,::-1,:,:,:]
    
    aux2 = eigvals[int(sx/2):,:,:,:]
    eigvals[:int(sx/2),:,:,:] = aux2
    eigvals[int(sx/2):,:,:,:] = aux2[::-1,::-1,:,:]

    return eigvals,eigvects

# Creates V
def sintV(l1, l2, l3, n, r, a, l1_back, l2_back, l3_back, background=True):
    import numpy as np
    
    b1 = 0

    x,y,z = np.mgrid[0:n,0:n,0:n]
    R = a*x + b1 - y # equacao da reta (voxels pertencentes a reta)
    R = np.where(((R**2) + (z-(n/2))**2)**0.5 < r) # distancia
    i,j,k = R
    vaux = np.zeros((n,n,n))
    vaux[i,j,k] = 1 # voxels pertencentes a metade do V
    
    v = np.zeros((n,n,n))
    v[:int(n/2),:,:] = vaux[:int(n/2),:,:]
    vaux2 = vaux[::-1,:,:]
    v[int(n/2):,:,:] = vaux2[int(n/2):,:,:] # voxels pertencentes ao V (1 onde pertence e 0 é background)

    # Inicializa os autovetores do background
    eigvectsback = np.zeros(np.append(x.shape,(3,3)))  # <----
    if background:
        eigvectsback[:,:,:,2,0] = 1 # coloca os vetores ortogonais
        eigvectsback[:,:,:,1,1] = 1
        eigvectsback[:,:,:,0,2] = 1

    # autovetores do V
    # ev1
    ev1_1 = np.array([0,a,1])/(a**2 + 1)**0.5
    ev1_2 = np.array([0,a,-1])/(a**2 + 1)**0.5
    
    eigvectV = np.zeros(np.append(x.shape,(3,3)))
    eigvectV[:int(n/2),:,:,0,0] = ev1_1[2]*v[:int(n/2),:,:]
    eigvectV[int(n/2):,:,:,0,0] = ev1_2[2]*v[int(n/2):,:,:]
    eigvectV[:int(n/2),:,:,1,0] = ev1_1[1]*v[:int(n/2),:,:]
    eigvectV[int(n/2):,:,:,1,0] = ev1_2[1]*v[int(n/2):,:,:]

    # ev2
    eigvectV[:,:,:,1,1] = -eigvectV[:,:,:,0,0]
    eigvectV[:,:,:,0,1] = eigvectV[:,:,:,1,0]

    # ev3
    eigvectV[:,:,:,2,2] = 1

    eigvect = eigvectV*v[..., None, None] + (1-v[..., None, None])*eigvectsback

    #Autovalores, diferentes para o V e para o background
    eigvalsV = np.ones(np.append(x.shape,3))
    eigvalsV[...,0] *= l1
    eigvalsV[...,1] *= l2
    eigvalsV[...,2] *= l3
    eigvals_back = np.ones(eigvalsV.shape)
    eigvals_back[...,0] *= l1_back
    eigvals_back[...,1] *= l2_back
    eigvals_back[...,2] *= l3_back
    eigvals = eigvalsV*v[..., None] + (1-v[..., None])*eigvals_back

    return eigvals,eigvect