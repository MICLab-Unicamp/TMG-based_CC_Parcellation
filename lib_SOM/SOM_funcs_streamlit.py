import numpy as np 
from sklearn_som.som import SOM
from skimage.measure import label
from skimage.morphology import binary_dilation
from lib_TMG import TMGUtil as util

def SOM_parc(tmg, mask, evals, fa, sb_msp = 0, n_labels = 5, epochs=10, n_rep=100):

    ori_shape = mask.shape

    # Getting the MSP slice
    msp_slice = get_msp_slice(evals, fa)
    # If you want a slice close to the MSP
    msp_slice += sb_msp

    # Taking the mask only at the MSP
    mask_msp = mask[msp_slice]
    # Cutting a bounding box around the CC
    mask_cut, pos_cut = BB_CC_MSP(mask_msp)

    # Getting positions of each pixel belonging to the mask
    i,j = np.indices(mask_cut.shape)
    i_mask = i[mask_cut==1]
    j_mask = j[mask_cut==1]

    # Normalizing positions from 0 to 1
    i_mask_n = (i_mask - i_mask.min())/(i_mask.max() - i_mask.min())
    j_mask_n = (j_mask - j_mask.min())/(j_mask.max() - j_mask.min())

    # Combining positions into a single array
    i_mask_n = np.expand_dims(i_mask_n, axis=1)
    j_mask_n = np.expand_dims(j_mask_n, axis=1)
    pos_n = np.append(i_mask_n, j_mask_n, axis=1)

    # Taking only the MSP slice of the TMG
    tmg_msp = tmg[msp_slice]
    # Cutting according to the bounding box
    tmg_cut = tmg_msp[pos_cut[0], pos_cut[1]]
    # Taking only mask pixels
    tmg_mask = tmg_cut[mask_cut==1]

    # Normalizing from 0 to 1
    tmg_i_n = (tmg_mask - tmg_mask.min())/(tmg_mask.max() - tmg_mask.min())
    tmg_i_n = np.expand_dims(tmg_i_n, axis=1)

    # Joining TMG data with pixel position information
    final_data = np.append(tmg_i_n, pos_n, axis=1)

    # Three features/dimensions: one for TMG intensities and two for pixel positions
    dim = 3

    # Array to store the count of labels each pixel received across repetitions
    count_labels = np.zeros(np.append(mask_cut.shape, n_labels+1))

    for r in range(n_rep):
        # Generate the SOM from the image data vector (containing intensities and pixel positions)
        som1 = SOM(m=n_labels, n=1, dim=dim, max_iter=epochs*final_data.shape[0])
        som1.fit(final_data, epochs=epochs)
        # Get the prediction for the same image used to fit
        predictions1 = som1.predict(final_data)

        # Reorganize predictions into the bounding box dimensions
        first_img = np.ones(mask_cut.shape)*-1
        first_img[i_mask, j_mask] = predictions1
        # Redefine labels according to connected components
        label_fi = label(first_img, background=-1, connectivity=1)

        # Function to merge the smallest connected component into the most prevalent adjacent component until only 5 components remain
        img_con = merge_small_comp(label_fi.copy(), n_labels)

        # Again, relabel according to connected components after reduction
        final_img = label(img_con, background=0, connectivity=1)

        # For each train/predict repetition, count the label assigned to each pixel
        for u in np.unique(final_img):
            x = np.where(final_img == u, 1, 0)
            count_labels[...,u] += x

    # Finally, each pixel is assigned the label with the most "votes" after repetitions
    final_parc_bb = label(np.argmax(count_labels, axis=-1), background=0, connectivity=1)
    # Uncertainty
    temp_uncert = np.max(count_labels, axis=-1)

    # Function to merge the smallest connected component into the most prevalent adjacent component until only 5 components remain
    final_parc_bb_con = merge_small_comp(final_parc_bb.copy(), n_labels)

    # Again, relabel according to connected components after reduction
    final_parc_bb_con = label(final_parc_bb_con, background=0, connectivity=1)

    # Bring the parcellation back to original image dimensions
    final_parc = np.zeros(ori_shape, dtype=mask_cut.dtype)
    final_parc[msp_slice][pos_cut[0], pos_cut[1]] = final_parc_bb_con
    # Uncertainty
    final_uncert = np.zeros(ori_shape, dtype=float)
    final_uncert[msp_slice][pos_cut[0], pos_cut[1]] = temp_uncert*mask_cut/n_rep

    return final_parc, final_uncert

def get_msp_slice(evals, fa):
    import numpy as np

    MASK = (evals[...,0] > 0)
    MASKcount = MASK.sum(axis=2).sum(axis=1)
    FAmean = fa.mean(axis=2).mean(axis=1)
    FAmean[MASKcount <= 0.90*MASKcount.max()] = 1

    return np.argmin(FAmean)

def BB_CC_MSP(mask, get_min_max=False, pad=0, bin=True):
    # Copy the input array so as not to modify it
    mask_norm = mask.copy()

    if bin:
        # Ensure the mask is binary
        mask_norm[mask_norm != 0] = 1

        # Get indices of voxels of interest
        roi_ind = np.where(mask_norm == 1)
    else:
        # Get indices of voxels of interest
        roi_ind = np.where(mask_norm > 0)
    
    # Get min/max per dimension (mask boundaries)
    min_x = np.min(roi_ind[0]) - pad
    max_x = np.max(roi_ind[0]) + pad
    min_y = np.min(roi_ind[1]) - pad
    max_y = np.max(roi_ind[1]) + pad
    slice_x = slice(min_x,max_x+1)
    slice_y = slice(min_y,max_y+1)

    # Bounding box around the mask to avoid unnecessary calculations
    mask_cut = np.int8(mask_norm[slice_x,slice_y])

    if get_min_max:
        return mask_cut, [slice_x,slice_y], [min_x,max_x,min_y,max_y]

    return mask_cut, [slice_x,slice_y]

def merge_small_comp(img_con, n_labels):
    es = np.zeros((3,3))
    es[1] = 1
    es[:,1] = 1
    labels, conts = np.unique(img_con[img_con!=0], return_counts=True)
    while (len(labels) > n_labels):
        smaller_label = labels[np.argmin(conts)]
        
        temp = np.zeros(img_con.shape)
        temp[img_con == smaller_label] = 1
        temp2 = binary_dilation(temp, footprint=es)
        temp = temp2-temp
        
        temp2 = img_con[temp != 0]
        temp2 = temp2[temp2 != 0]

        labels_v, conts_v = np.unique(temp2, return_counts=True)
        new_label = labels_v[np.argmax(conts_v)]

        img_con[img_con == smaller_label] = new_label
        labels, conts = np.unique(img_con[img_con!=0], return_counts=True)

    return img_con