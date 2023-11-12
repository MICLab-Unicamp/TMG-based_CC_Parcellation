def TensorCalc(eigvals,eigvects):
    import numpy as np

    tensors = np.empty(np.append(eigvects.shape[0:4],6), dtype=eigvects.dtype) #,eigvects.shape[3],eigvects.shape[4]))
    # Retorna as componentes na seguinte ordem: txx,tyy,tzz,txy,txz,tyz
    tensors[...,0] = eigvals[...,0]*eigvects[...,0,0]**2+eigvals[...,1]*eigvects[...,0,1]**2+eigvals[...,2]*eigvects[...,0,2]**2
    tensors[...,1] = eigvals[...,0]*eigvects[...,1,0]**2+eigvals[...,1]*eigvects[...,1,1]**2+eigvals[...,2]*eigvects[...,1,2]**2
    tensors[...,2] = eigvals[...,0]*eigvects[...,2,0]**2+eigvals[...,1]*eigvects[...,2,1]**2+eigvals[...,2]*eigvects[...,2,2]**2
    tensors[...,3] = eigvals[...,0]*eigvects[...,0,0]*eigvects[...,1,0]+eigvals[...,1]*eigvects[...,0,1]*eigvects[...,1,1]+eigvals[...,2]*eigvects[...,0,2]*eigvects[...,1,2]
    tensors[...,4] = eigvals[...,0]*eigvects[...,0,0]*eigvects[...,2,0]+eigvals[...,1]*eigvects[...,0,1]*eigvects[...,2,1]+eigvals[...,2]*eigvects[...,0,2]*eigvects[...,2,2]
    tensors[...,5] = eigvals[...,0]*eigvects[...,1,0]*eigvects[...,2,0]+eigvals[...,1]*eigvects[...,1,1]*eigvects[...,2,1]+eigvals[...,2]*eigvects[...,1,2]*eigvects[...,2,2]

    return tensors

def TensorToComponents(tensors):

    txx = tensors[...,0]
    tyy = tensors[...,1]
    tzz = tensors[...,2]
    txy = tensors[...,3]
    txz = tensors[...,4]
    tyz = tensors[...,5]

    return txx,tyy,tzz,txy,txz,tyz

def FullTensorCalc(eigvals,eigvects):
    import numpy as np

    tensors = np.empty(np.append(eigvects.shape[0:4],(3,3)), dtype=eigvects.dtype)
    tensors[...,0,0] = eigvals[...,0]*eigvects[...,0,0]**2+eigvals[...,1]*eigvects[...,0,1]**2+eigvals[...,2]*eigvects[...,0,2]**2
    tensors[...,1,1] = eigvals[...,0]*eigvects[...,1,0]**2+eigvals[...,1]*eigvects[...,1,1]**2+eigvals[...,2]*eigvects[...,1,2]**2
    tensors[...,2,2] = eigvals[...,0]*eigvects[...,2,0]**2+eigvals[...,1]*eigvects[...,2,1]**2+eigvals[...,2]*eigvects[...,2,2]**2
    tensors[...,0,1] = tensors[...,1,0] = eigvals[...,0]*eigvects[...,0,0]*eigvects[...,1,0]+eigvals[...,1]*eigvects[...,0,1]*eigvects[...,1,1]+eigvals[...,2]*eigvects[...,0,2]*eigvects[...,1,2]
    tensors[...,2,0] = tensors[...,0,2] = eigvals[...,0]*eigvects[...,0,0]*eigvects[...,2,0]+eigvals[...,1]*eigvects[...,0,1]*eigvects[...,2,1]+eigvals[...,2]*eigvects[...,0,2]*eigvects[...,2,2]
    tensors[...,2,1] = tensors[...,1,2] = eigvals[...,0]*eigvects[...,1,0]*eigvects[...,2,0]+eigvals[...,1]*eigvects[...,1,1]*eigvects[...,2,1]+eigvals[...,2]*eigvects[...,1,2]*eigvects[...,2,2]

    return tensors

def EigvalsToComponents(eigvals):

    lambda1 = eigvals[...,0]
    lambda2 = eigvals[...,1]
    lambda3 = eigvals[...,2]

    return lambda1,lambda2,lambda3

def EigvectsToComponents(eigvects):

    e1x = eigvects[...,0,0]
    e1y = eigvects[...,1,0]
    e1z = eigvects[...,2,0]
    e2x = eigvects[...,0,1]
    e2y = eigvects[...,1,1]
    e2z = eigvects[...,2,1]
    e3x = eigvects[...,0,2]
    e3y = eigvects[...,1,2]
    e3z = eigvects[...,2,2]

    return e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z