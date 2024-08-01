def cut_imgs_mask(img, mask, get_min_max=False, pad=0, bin=True):
    import numpy as np 

    shape_ori = mask.shape
    # Cópia das matrizes de entrada, para não alterá-las
    mask_norm = mask.copy()

    if bin:
        # Garante que a máscara é binária
        mask_norm[mask_norm != 0] = 1

        # Obtém os índices dos voxels de interesse
        roi_ind = np.where(mask_norm == 1)
    else:
        # Obtém os índices dos voxels de interesse
        roi_ind = np.where(mask_norm > 0)
    
    # Pega os máximos e mínimos em cada dimensão (extremidades da máscara)
    min_x = np.min(roi_ind[0]) - pad
    max_x = np.max(roi_ind[0]) + pad
    min_y = np.min(roi_ind[1]) - pad
    max_y = np.max(roi_ind[1]) + pad
    min_z = np.min(roi_ind[2]) - pad
    max_z = np.max(roi_ind[2]) + pad
    slice_x = slice(min_x,max_x+1)
    slice_y = slice(min_y,max_y+1)
    slice_z = slice(min_z,max_z+1)

    # Bounding box em torno da máscara para evitar cálculos desnecessários
    img_cut = img.copy()
    img_cut = img_cut[slice_x,slice_y,slice_z]
    mask_cut = np.int8(mask_norm[slice_x,slice_y,slice_z])

    if get_min_max:
        return img_cut, mask_cut, [slice_x,slice_y,slice_z], [min_x,max_x,min_y,max_y,min_z,max_z]

    return img_cut, mask_cut, [slice_x,slice_y,slice_z]

def vis_imgs_2D(imgs, plane, slice_number, n_col, titles=None, cmap='gray', f_m_s_c=5, f_m_s_r=5, file_name = None, return_fig = False):
    import matplotlib.pyplot as plt
    import numpy as np

    #Configurando os plots
    if len(imgs)%n_col == 0:
        n_lin = len(imgs)//n_col
    else:
        n_lin = len(imgs)//n_col+1
    fig, axs = plt.subplots(n_lin, n_col)
    fig.set_facecolor([0.9, 0.9, 0.9])
    fig.set_size_inches(f_m_s_c*n_col, f_m_s_r*n_lin)

    for n,img in enumerate(imgs):
        if isinstance(slice_number, int):
            sn = slice_number
        else:
            sn = slice_number[n]

        if n_lin == 1 or n_col == 1:
            plt_pos = n
        else:
            plt_pos = (n//n_col,n-(n//n_col)*n_col)

        if n_lin == 1 and n_col == 1:
            if plane == 'xy':
                axs.imshow(np.rot90(img[:,:,sn][::-1]), cmap=cmap)

            if plane == 'yz':
                axs.imshow(np.rot90(img[sn][::-1]), cmap=cmap)

            if plane == 'xz':
                axs.imshow(np.rot90(img[:,sn][::-1]), cmap=cmap)

            if titles is not None:
                axs.set_title(titles[n])
            axs.axis('off')
        else:
            if plane == 'xy':
                axs[plt_pos].imshow(np.rot90(img[:,:,sn][::-1]), cmap=cmap)

            if plane == 'yz':
                axs[plt_pos].imshow(np.rot90(img[sn][::-1]), cmap=cmap)

            if plane == 'xz':
                axs[plt_pos].imshow(np.rot90(img[:,sn][::-1]), cmap=cmap)

            if titles is not None:
                axs[plt_pos].set_title(titles[n])
            axs[plt_pos].axis('off')

    if return_fig:
        return fig

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, format='png')

    return

def get_msp_slice(path_fa, path_evals):
    from dipy.io.image import load_nifti
    import numpy as np

    fa,_ = load_nifti(path_fa)
    evals,_ = load_nifti(path_evals)

    MASK = (evals[...,0] > 0)
    MASKcount = MASK.sum(axis=2).sum(axis=1)
    FAmean = fa.mean(axis=2).mean(axis=1)
    FAmean[MASKcount <= 0.90*MASKcount.max()] = 1

    return (np.argmin(FAmean), FAmean)