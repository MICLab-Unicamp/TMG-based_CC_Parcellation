def TMG(eigvals,eigvects,similarity,neighborhood=6,nbh_info=None,mask=None):
    import numpy as np 
    from lib_TMG import TMGMetrics as mtc
    from lib_TMG import TMGSE as se

    dict_nbh = {2: se.two_connected, 4: se.four_connected, 6: se.six_connected, 8: se.eight_connected}

    # Copy input arrays so as not to modify them
    eigvalslin = eigvals.copy()
    eigvectslin = eigvects.copy()

    if mask is not None:
        # Copy input arrays so as not to modify them
        mask_norm = mask.copy()
        # Ensure the mask is binary
        mask_norm[mask_norm != 0] = 1

        # Get indices of voxels of interest
        roi_ind = np.where(mask_norm == 1)
        # Get min/max per dimension (mask boundaries)
        min_x = np.min(roi_ind[0])
        max_x = np.max(roi_ind[0])
        min_y = np.min(roi_ind[1])
        max_y = np.max(roi_ind[1])
        min_z = np.min(roi_ind[2])
        max_z = np.max(roi_ind[2])

        # Bounding box around the mask to avoid unnecessary calculations
        eigvalslin = eigvalslin[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
        eigvectslin = eigvectslin[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
        masccut = np.int8(mask_norm[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1])

        # Zero background voxels to avoid unnecessary calculations
        masklin = np.expand_dims(masccut, axis = -1)
        eigvalslin = eigvalslin*masklin

        masklin = np.expand_dims(masklin, axis = -1)
        eigvectslin = eigvectslin*masklin

        # Flatten the matrix
        masklin.shape = np.prod(masklin.shape)

    shape = eigvectslin.shape

    # Transform the first 3 dimensions into one
    eigvalslin.shape = np.append(np.prod(eigvalslin.shape[0:3]),eigvalslin.shape[3])
    eigvectslin.shape = np.append(np.prod(eigvectslin.shape[0:3]),eigvectslin.shape[3::])
    
    # For each voxel, calculate the indices of voxels in its neighborhood
    indices = dict_nbh[neighborhood](shape, nbh_info)

    # Select
    eigvalsb = eigvalslin[indices,:]
    eigvectsb = eigvectslin[indices,:,:]

    if mask is not None:
        maskb = masklin[indices]
    else:
        maskb = None

    # Compute distances
    distance = mtc.tensorialSimilarityMeasures(eigvalsb,eigvectsb,similarity,neighborhood,mask=maskb)

    if mask is not None:
        # Multiply distance by the mask so background voxels are zero
        distance = distance*np.expand_dims(masccut, axis = -1)
    
    # Compute the actual TMG
    if (similarity == 'prod'):
        # For similarity metrics, take the minimum across distances
        if mask is not None:
            img = np.zeros(mask.shape, dtype=distance.dtype)
            img[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1] = distance.min(axis=-1)
            # Compute the negative of prod (inside the mask only) to match other metrics
            img[mask_norm == 1] = img[mask_norm == 1].max()-img[mask_norm == 1]
        else:
            img = distance.min(axis=-1)
            img = img.max() - img
        
    else:
        # For dissimilarity metrics, take the maximum across distances
        if mask is not None:
            img = np.zeros(mask.shape, dtype=distance.dtype)
            img[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1] = distance.max(axis=-1)
        else:
            img = distance.max(axis=-1)

    return img
