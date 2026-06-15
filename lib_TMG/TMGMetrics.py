# Functions that define the distance metrics and calculate the distance between each pair of voxels in a neighborhood

def tensorialSimilarityMeasures(eigvals,eigvects,similarity,neighborhood,mode='default',mask=None):
    import numpy as np 
    import os
    
    # Calculate all pairwise distances, even the redundant ones (such as 0-1 and 1-0)
    if mode == 'fullc':
        j,i = np.mgrid[0:eigvals.shape[-2],0:eigvals.shape[-2]]
    # Use the file 'dict_comps.npy' to avoid calculating redundant distances
    else:
        dict_comps = np.load(os.path.join(os.path.dirname(__file__),'dict_comps.npy'), allow_pickle=True).tolist()
        i = dict_comps[neighborhood]['i']
        j = dict_comps[neighborhood]['j']

    # Calculate the distances using the selected metric
    distance = DistanceCalc(eigvals, eigvects, similarity, i, j)

    if mask is not None:
        # If the comparison involves any background voxel, its value is zero
        mask_dist = mask[...,i]*mask[...,j]

        # In the case of prod, those comparisons would incorrectly be zero, so add (1-mask_dist) to add one to those results
        # Thus, when taking the minimum, those problematic comparisons are usually ignored
        if similarity == 'prod':
            distance = distance*mask_dist + (1-mask_dist)
        # Following the same idea for dissimilarity metrics, simply multiply distance*mask_dist
        # Thus comparisons involving background voxels are zeroed and ignored when taking the maximum
        else:
            distance = distance*mask_dist

    return distance

def DistanceCalc(eigvals, eigvects, similarity, i, j):
    import numpy as np
    from lib_TMG import TMGManipulation as ut

    distance = None

    # Dot Product (prod)
    if (similarity == 'prod'):
        # evecs that are all zero are changed to nan
        eigvects[np.where(np.abs(eigvects).sum(-1).sum(-1)==0)] = np.nan
        e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z = ut.EigvectsToComponents(eigvects)
        # Then when the distance for those voxels is computed it yields nan
        distance = np.abs(e1x[...,i]*e1x[...,j]+e1y[...,i]*e1y[...,j]+e1z[...,i]*e1z[...,j])
        # Replacing nan values with 1 so they are ignored when taking the minimum, unless all are problematic
        distance[np.isnan(distance)] = 1.0

    # Frobenius Norm (frob)
    if (similarity == 'frob'):
        tensors = ut.TensorCalc(eigvals,eigvects)
        txx,tyy,tzz,txy,txz,tyz = ut.TensorToComponents(tensors)
        txxr = (txx[...,i]-txx[...,j])**2
        tyyr = (tyy[...,i]-tyy[...,j])**2
        tzzr = (tzz[...,i]-tzz[...,j])**2
        txyr = (txy[...,i]-txy[...,j])**2
        txzr = (txz[...,i]-txz[...,j])**2
        tyzr = (tyz[...,i]-tyz[...,j])**2
        distance = (txxr+tyyr+tzzr+2*txyr+2*txzr+2*tyzr)**0.5

    # J-Divergence (Jdiv)
    if (similarity == 'Jdiv'):
        tensors = ut.TensorCalc(eigvals,eigvects)
        txx,tyy,tzz,txy,txz,tyz = ut.TensorToComponents(tensors)
        
        eigvalsi = eigvals.copy()
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

    # Log-Euclidean distance (logE)
    if (similarity == 'logE'):
        log_tensors = ut.TensorCalc(np.log(100*np.maximum(eigvals,0)+1),eigvects)
        txx,tyy,tzz,txy,txz,tyz = ut.TensorToComponents(log_tensors)
        txxr = (txx[...,i]-txx[...,j])**2
        tyyr = (tyy[...,i]-tyy[...,j])**2
        tzzr = (tzz[...,i]-tzz[...,j])**2
        txyr = (txy[...,i]-txy[...,j])**2
        txzr = (txz[...,i]-txz[...,j])**2
        tyzr = (tyz[...,i]-tyz[...,j])**2
        distance = (txxr+tyyr+tzzr+2*txyr+2*txzr+2*tyzr)**0.5

    return distance