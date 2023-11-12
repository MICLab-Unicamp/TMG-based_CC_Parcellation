def tensorialSimilarityMeasures(eigvals,eigvects,similarity,neighborhood,mode='default',mask=None):
    import numpy as np
    import os

    # dict_comps = {}
    # dict_comps[2] = {'i': [1,2,  2], 
    #                  'j': [0,0,  1]}
    # dict_comps[4] = {'i': [1,2,3,4,  2,3,4,  3,4,  4], 
    #                  'j': [0,0,0,0,  1,1,1,  2,2,  3]}
    # dict_comps[6] = {'i': [1,2,3,4,5,6,  2,3,4,5,6,  3,4,5,6,  4,5,6,  5,6,  6], 
    #                  'j': [0,0,0,0,0,0,  1,1,1,1,1,  2,2,2,2,  3,3,3,  4,4,  5]}
    # dict_comps[8] = {'i': [1,2,3,4,5,6,7,8,  2,3,4,5,6,7,8,  3,4,5,6,7,8,  4,5,6,7,8,  5,6,7,8,  6,7,8,  7,8,  8], 
    #                  'j': [0,0,0,0,0,0,0,0,  1,1,1,1,1,1,1,  2,2,2,2,2,2,  3,3,3,3,3,  4,4,4,4,  5,5,5,  6,6,  7]}
    # np.save('dict_comps', dict_comps, allow_pickle=True)
    
    if mode == 'fullc':
        j,i = np.mgrid[0:eigvals.shape[-2],0:eigvals.shape[-2]]
    else:
        dict_comps = np.load(os.path.join(os.path.dirname(__file__),'dict_comps.npy'), allow_pickle=True).tolist()
        i = dict_comps[neighborhood]['i']
        j = dict_comps[neighborhood]['j']

    distance = DistanceCalc(eigvals, eigvects, similarity, i, j)

    if mask is not None:
        #Se a comparação considerar algum voxel de fundo, o valor associado a ela é zero
        mask_dist = mask[...,i]*mask[...,j]

        #No caso do prod, essas comparações teriam erroneamente o valor zero, então faço a soma com (1-mask_dist) para adicionar um ao resultado dessas comparações
        #Assim, quando pegar o mínimo, essas comparações problemáticas serão geralmente desconsideradas
        if similarity == 'prod':
            distance = distance*mask_dist + (1-mask_dist)
        #Seguindo a mesma ideia para as métricas de dissimilaridade, simplesmente multiplico distance*mask_dist
        #Assim, comparações considerando voxels de fundo são zeradas e ignoradas ao pegar o máximo
        else:
            distance = distance*mask_dist

    return distance

def DistanceCalc(eigvals, eigvects, similarity, i, j):
    import numpy as np
    from lib_TMG import TMGManipulation as ut

    distance = None

    # produto escalar
    if (similarity == 'prod'):
        #evecs que são todos zero modificados para nan
        eigvects[np.where(np.abs(eigvects).sum(-1).sum(-1)==0)] = np.nan
        e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z = ut.EigvectsToComponents(eigvects)
        #Daí quando a distância considerando esses voxels é calculada ela resulta em nan
        distance = np.abs(e1x[...,i]*e1x[...,j]+e1y[...,i]*e1y[...,j]+e1z[...,i]*e1z[...,j])
        #Substituindo os valores nan por 1, quando pegar o mínimo, esses valores são ignorados, a não ser que todos sejam problematicos
        distance[np.isnan(distance)] = 1.0

    # norma de frobenius
    if (similarity == 'frob'):
        tensors = ut.TensorCalc(eigvals,eigvects)
        txx,tyy,tzz,txy,txz,tyz = ut.TensorToComponents(tensors)
        txxr = (txx[...,i]-txx[...,j])**2
        tyyr = (tyy[...,i]-tyy[...,j])**2
        tzzr = (tzz[...,i]-tzz[...,j])**2
        distance = (txxr+tyyr+tzzr)**0.5

    # norma de J-Divergence
    if (similarity == 'Jdiv'):
        tensors = ut.TensorCalc(eigvals,eigvects)
        txx,tyy,tzz,txy,txz,tyz = ut.TensorToComponents(tensors)
        
        eigvalsi = eigvals.copy()
        #Como tratar eigvals negativos?
        eigvalsi = np.maximum(eigvalsi, 0.0)
        ind = eigvalsi != 0.0
        eigvalsi[ind] = eigvalsi[ind]**-1
        tensorsi = ut.TensorCalc(eigvalsi,eigvects)
        
        txxi,tyyi,tzzi,txyi,txzi,tyzi = ut.TensorToComponents(tensorsi)
        temp1a = txxi[...,i]*txx[...,j]+txyi[...,i]*txy[...,j]+txzi[...,i]*txz[...,j]
        temp1b = txxi[...,j]*txx[...,i]+txyi[...,j]*txy[...,i]+txzi[...,j]*txz[...,i]
        temp2a = txyi[...,i]*txy[...,j]+tyyi[...,i]*tyy[...,j]+tyzi[...,i]*tyz[...,j]
        temp2b = txyi[...,j]*txy[...,i]+tyyi[...,j]*tyy[...,i]+tyzi[...,j]*tyz[...,i]
        temp3a = txzi[...,i]*txz[...,j]+tyzi[...,i]*tyz[...,j]+tzzi[...,i]*tzz[...,j]
        temp3b = txzi[...,j]*txz[...,i]+tyzi[...,j]*tyz[...,i]+tzzi[...,j]*tzz[...,i]
        distance = 0.5*(np.maximum(temp1a+temp1b+temp2a+temp2b+temp3a+temp3b,6.0)-6.0)**0.5

    # norma de log-Euclidean correta (aproximação aplicada aos autovalores)
    if (similarity == 'logE'):
        #Como tratar eigvals negativos pra evitar problema no log? Por enquanto tô zerando valores negativos...
        log_tensors = ut.TensorCalc(np.log(100*np.maximum(eigvals,0)+1),eigvects)
        txx,tyy,tzz,txy,txz,tyz = ut.TensorToComponents(log_tensors)
        txxr = (txx[...,i]-txx[...,j])**2
        tyyr = (tyy[...,i]-tyy[...,j])**2
        tzzr = (tzz[...,i]-tzz[...,j])**2

        distance = (txxr+tyyr+tzzr)**0.5

    return distance