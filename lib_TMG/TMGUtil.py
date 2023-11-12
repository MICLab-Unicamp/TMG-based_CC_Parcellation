def cut_imgs_mask(img, mask, get_min_max=False, pad=0, bin=True):
    import numpy as np

    shape_ori = mask.shape
    # Cópia das matrizes de entrada, para não alterá-las
    mask_norm = mask.copy()
    #print(np.unique(mask))
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

    #print(img_cut.shape)

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

def calc_hist_min_max(mode, img, mask, nbins):
    import numpy as np

    if mode == 'default':
        img_hist = img[np.where(mask == 1)]
        min = img_hist.min()
        max = img_hist.max()
        h, bin_edges = np.histogram(img_hist, nbins, (min, max))
        return h, bin_edges, min, max

    elif mode == 'triple':
        img_hist = img[...,0,:]
        img_hist = img_hist[np.where(mask == 1)]
        min = [img_hist[...,0].min(), img_hist[...,1].min(), img_hist[...,2].min()]
        max = [img_hist[...,0].max(), img_hist[...,1].max(), img_hist[...,2].max()]
        h_1, bin_edges_1 = np.histogram(img_hist[...,0], nbins, (min[0], max[0]))
        h_2, bin_edges_2 = np.histogram(img_hist[...,1], nbins, (min[1], max[1]))
        h_3, bin_edges_3 = np.histogram(img_hist[...,2], nbins, (min[2], max[2]))
        return [h_1, h_2, h_3], [bin_edges_1, bin_edges_2, bin_edges_3], min, max

#Data/{dataset}/{sid}/{roi}/nbh/
def info_tmg_data(datasets, nbhs, sids=['*'], rois=[None], metrics=['prod', 'frob', 'Jdiv', 'logE'], nbins=255):
    import glob
    import os
    from dipy.io.image import load_nifti
    import numpy as np

    dict_infos = {'File_path': [], 'Image': [], 'Mask': [], 'Min': [], 'Max': [], 'Hist': [], 'Bin_edges': [], 'MSP_slice': []}

    path = 'Data/'
    for d in datasets:
        path1 = path + d + '/'
        for s in sids:
            path2 = path1 + s + '/'
            for r in rois:

                if r is None:
                    dir_tmg = 'TMG/'
                    path3 = path2 + dir_tmg
                    mask_name = 'brain_mask.nii.gz'
                else:
                    dir_tmg = f'TMG_CCmm/'
                    path3 = path2 + dir_tmg
                    mask_name = f'{r[0:2]}_mask_{r[2:]}.nii.gz'

                for n in nbhs:
                    path4 = path3 + f'n_{n}/'

                    for m in metrics:

                        file_name = f'{m}.nii.gz'

                        for p in glob.glob(path4+file_name):

                            p = os.path.normpath(p)
                            p_split = p.split(os.sep)
                            db = p_split[1]
                            sid = p_split[2]
                            print(p)
                            img,_ = load_nifti(p)

                            dict_infos['File_path'].append(p)
                            if isinstance(n, int):
                                dict_infos['Image'].append(img)
                            else:
                                dict_infos['Image'].append(img[...,0,:])

                            path_mask = f'Data/{db}/{sid}/'
                            
                            mask,_ = load_nifti(path_mask+mask_name)
                            mask[mask != 0] = 1

                            dict_infos['Mask'].append(mask)
                            
                            path_fa = f'Data/{db}/{sid}/FA.nii.gz'

                            msp_slice = get_msp_slice(path_fa)

                            dict_infos['MSP_slice'].append(msp_slice)

                            if isinstance(n,int):
                                mode = 'default'
                            else:
                                mode = 'triple'

                            h, bin_edges, min, max = calc_hist_min_max(mode, img, mask, nbins)

                            dict_infos['Min'].append(min)
                            dict_infos['Max'].append(max)
                            dict_infos['Hist'].append(h)
                            dict_infos['Bin_edges'].append(bin_edges)

    return dict_infos

def vis_hist_tmg(infos, n_col=4, nbins=255):
    import matplotlib.pyplot as plt
    import numpy as np

    n_plots = len(infos['File_path'])

    if n_plots%n_col == 0:
        n_lin = n_plots//n_col
    else:
        n_lin = n_plots//n_col+1
    fig, axs = plt.subplots(n_lin, n_col)
    fig.set_size_inches(20, 5*n_lin)

    for n,i in enumerate(range(n_plots)):
        if (n_lin == 1) or (n_col==1):
            pos = n
        else:
            pos = (n//n_col,n-(n//n_col)*n_col)

        w = (infos['Max'][i] - infos['Min'][i])/nbins
        #bin_centers = infos['Bin_edges'][i][2:]-(w/2)
        bin_centers = infos['Bin_edges'][i][1:]-(w/2)
        #axs[n//n_col,n-(n//n_col)*n_col].bar(bin_centers, (infos['Hist'][i][1:]/np.max(infos['Hist'][i][1:])), width=w)
        axs[pos].bar(bin_centers, (infos['Hist'][i]), width=w)
        #axs[n//n_col,n-(n//n_col)*n_col].plot(infos['CDF_bins'][i], infos['CDF'][i], c='r')
        axs[pos].set_title(str(infos['File_path'][i][8:-7]) + '\n' + str(infos['Min'][i]) + ' ~ ' + str(infos['Max'][i]))
    plt.show()

    return

def comp_hist_tmg(infos, nbins=255, lim=None):
    import matplotlib.pyplot as plt
    import numpy as np

    n_plots = len(infos['File_path'])

    for n,i in enumerate(range(n_plots)):
        w = (infos['Max'][i] - infos['Min'][i])/nbins
        bin_centers = infos['Bin_edges'][i][1:]-(w/2)
        plt.plot(bin_centers, infos['Hist'][i])

    if lim is not None:
        plt.xlim(lim)
    plt.show()

    return

def vis_hist_ttmg(infos, n_col=3, nbins=255):
    import matplotlib.pyplot as plt
    import numpy as np

    n_plots = len(infos['File_path'])*3

    if n_plots%n_col == 0:
        n_lin = n_plots//n_col
    else:
        n_lin = n_plots//n_col+1
    fig, axs = plt.subplots(n_lin, n_col)
    fig.set_size_inches(20, 5*n_lin)

    for n,i in enumerate(range(len(infos['File_path']))):
        #Para cada canal
        for j in range(3):
            w = (infos['Max'][i][j] - infos['Min'][i][j])/nbins
            bin_centers = infos['Bin_edges'][i][j][1:]-(w/2)
            if n_lin == 1:
                axs[(n-1)*3+j].bar(bin_centers, (infos['Hist'][i][j]), width=w)
                #axs[(n-1)*3+j].plot(infos['CDF_bins'][i][j], infos['CDF'][i][j], c='r')
                axs[(n-1)*3+j].set_title(str(infos['File_path'][i][8:-7]) + ' - c' + str(j) + '\n' + str(infos['Min'][i]) + ' ~ ' + str(infos['Max'][i]))
            else:
                axs[((n-1)*3+j)//n_col,((n-1)*3+j)-(((n-1)*3+j)//n_col)*n_col].bar(bin_centers, (infos['Hist'][i][j]), width=w)
                #axs[((n-1)*3+j)//n_col,((n-1)*3+j)-(((n-1)*3+j)//n_col)*n_col].plot(infos['CDF_bins'][i][j], infos['CDF'][i][j], c='r')
                axs[((n-1)*3+j)//n_col,((n-1)*3+j)-(((n-1)*3+j)//n_col)*n_col].set_title(str(infos['File_path'][i][8:-7]) + ' - c' + str(j) + '\n' + str(infos['Min'][i]) + ' ~ ' + str(infos['Max'][i]))
    plt.show()

    return

def comp_hist_ttmg(infos, nbins=255, lim=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)

    n_plots = len(infos['File_path'])

    for i in range(n_plots):
        #Para cada canal
        for j in range(3):
            w = (infos['Max'][i][j] - infos['Min'][i][j])/nbins
            bin_centers = infos['Bin_edges'][i][j][1:]-(w/2)
            axs[j].plot(bin_centers, infos['Hist'][i][j])
            axs[j].set_title('c' + str(j))

        #w = (infos['Max'][i] - infos['Min'][i])/nbins
        #bin_centers = infos['Bin_edges'][i][1:]-(w/2)
        #plt.plot(bin_centers, infos['Hist'][i])

    if lim is not None:
        plt.xlim(lim)
    plt.show()

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