def TMG(eigvals,eigvects,similarity,neighborhood=6,nbh_info=None,mask=None):
    '''
    Inputs:
    
    eigvals ->      DTI eigenvalues following DIPY convention (shape = (x,y,z,3),
                    eigvals[...,0] = l1, eigvals[...,1] = l2, eigvals[...,1] = l3).
    
    eigvects ->     DTI eigenvectors following DIPY convention (shape = (x,y,z,3,3), eigenvectors are stored columnwise, i.e.,
                    the last dimension of the array defines the eigenvector).

    similarity ->   Define the metric used to calculate the TMG. The options are "prod", "frob", "logE", and "Jdiv".
                    For more information on the metrics, check the file "TMG Metrics Info and Examples on Synthetic Data.ipynb".

    neighborhood -> Define the neighborhood (structuring element) used to calculate the TMG. The options are 2, 4, 6, and 8.
                    To visualize the different options, check the file "TMG Neighborhood Visualization.ipynb".

    nbh_info ->     Define the orientation of the structuring element, not used for neighborhood = 6 (3D structuring element).
                    If neighborhood = 2, the options are "x", "y", and "z".
                    if neighborhood = 4 or 8, the options are "xy", "xz", and "yz".
                    To visualize the different options, check the file "TMG Neighborhood Visualization.ipynb".

    mask ->         If a mask is provided, the TMG will only be calculated inside of it.

    --------------------------------------------------

    Outputs:

    img ->          TMG 3D scalar image.
    '''

    import numpy as np 
    from lib_TMG import TMGMetrics as mtc
    from lib_TMG import TMGSE as se

    dict_nbh = {2: se.two_connected, 4: se.four_connected, 6: se.six_connected, 8: se.eight_connected}

    # Copy of input matrices
    eigvalslin = eigvals.copy()
    eigvectslin = eigvects.copy()

    # If an input mask is provided, the TMG will only be calculated inside of it
    if mask is not None:
        # Copy of input matrices
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
