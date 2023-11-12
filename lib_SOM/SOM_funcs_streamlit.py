import numpy as np
from sklearn_som.som import SOM
from skimage.measure import label
from skimage.morphology import binary_dilation
from lib_TMG import TMGUtil as util

def SOM_parc(tmg, mask, evals, fa, sb_msp = 0, n_labels = 5, epochs=10, n_rep=100):

    ori_shape = mask.shape

    #Obtendo o slice do MSP
    msp_slice = get_msp_slice(evals, fa)
    #Caso queira pegar alguma fatia próxima ao MSP
    msp_slice += sb_msp

    #Pegando a máscara apenas no MSP
    mask_msp = mask[msp_slice]
    #Cortando um bounding box em torno do CC
    mask_cut, pos_cut = BB_CC_MSP(mask_msp)

    #Obtendo as posições de cada pixel pertencente à máscara
    i,j = np.indices(mask_cut.shape)
    i_mask = i[mask_cut==1]
    j_mask = j[mask_cut==1]

    #Normalizando as posições de 0 a 1
    i_mask_n = (i_mask - i_mask.min())/(i_mask.max() - i_mask.min())
    j_mask_n = (j_mask - j_mask.min())/(j_mask.max() - j_mask.min())

    #Juntando as posições num único array
    i_mask_n = np.expand_dims(i_mask_n, axis=1)
    j_mask_n = np.expand_dims(j_mask_n, axis=1)
    pos_n = np.append(i_mask_n, j_mask_n, axis=1)

    #Pegando apenas o MSP slice do TMG
    tmg_msp = tmg[msp_slice]
    #Cortando de acordo com o bounding box
    tmg_cut = tmg_msp[pos_cut[0], pos_cut[1]]
    #Pegando apenas os pixels da máscara
    tmg_mask = tmg_cut[mask_cut==1]

    #Normalizando de 0 a 1
    tmg_i_n = (tmg_mask - tmg_mask.min())/(tmg_mask.max() - tmg_mask.min())
    tmg_i_n = np.expand_dims(tmg_i_n, axis=1)

    #Juntando as informações do TMG com as informações das posições de cada pixel
    final_data = np.append(tmg_i_n, pos_n, axis=1)

    #Três features/dimensões: uma para as intensidades do TMG e duas para as posições dos pixels
    dim = 3

    #Array para guardar a contagem de labels que cada pixel recebeu nas repetições
    count_labels = np.zeros(np.append(mask_cut.shape, n_labels+1))

    for r in range(n_rep):
        #Gera o SOM a partir do vetor de dados da imagem (contendo as intensidades e posições de cada pixel)
        som1 = SOM(m=n_labels, n=1, dim=dim, max_iter=epochs*final_data.shape[0])
        som1.fit(final_data, epochs=epochs)
        #Obtém a predição para a mesma imagem do fit
        predictions1 = som1.predict(final_data)

        #Reorganiza as predições obtidas para as dimensões do bounding box
        first_img = np.ones(mask_cut.shape)*-1
        first_img[i_mask, j_mask] = predictions1
        #Redefine as labels de acordo com os componentes conexos
        label_fi = label(first_img, background=-1, connectivity=1)

        #Função para fundir o menor componente conexo ao componente conexo mais presente em sua borda externa, até que restem apenas 5 componentes
        img_con = merge_small_comp(label_fi.copy(), n_labels)

        #Novamente, redefine as labels de acordo com os componentes conexos gerados após a redução
        final_img = label(img_con, background=0, connectivity=1)

        #Para cada repetição do "treino/predição", contabiliza a label definida para cada pixel
        for u in np.unique(final_img):
            x = np.where(final_img == u, 1, 0)
            count_labels[...,u] += x

    #Por fim, a cada pixel é definida a label com mais "votos" após as repetições
    final_parc_bb = label(np.argmax(count_labels, axis=-1), background=0, connectivity=1)
    #Incerteza
    temp_uncert = np.max(count_labels, axis=-1)

    #Função para fundir o menor componente conexo ao componente conexo mais presente em sua borda externa, até que restem apenas 5 componentes
    final_parc_bb_con = merge_small_comp(final_parc_bb.copy(), n_labels)

    #Novamente, redefine as labels de acordo com os componentes conexos gerados após a redução
    final_parc_bb_con = label(final_parc_bb_con, background=0, connectivity=1)

    #Traz o parcelamento para as dimensões da imagem original
    final_parc = np.zeros(ori_shape, dtype=mask_cut.dtype)
    final_parc[msp_slice][pos_cut[0], pos_cut[1]] = final_parc_bb_con
    #Incerteza
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
    # Cópia da matriz de entrada, para não alterá-la
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
    slice_x = slice(min_x,max_x+1)
    slice_y = slice(min_y,max_y+1)

    # Bounding box em torno da máscara para evitar cálculos desnecessários
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
        #print(labels, conts)
        smaller_label = labels[np.argmin(conts)]
        
        temp = np.zeros(img_con.shape)
        temp[img_con == smaller_label] = 1
        temp2 = binary_dilation(temp, footprint=es)
        temp = temp2-temp
        
        temp2 = img_con[temp != 0]
        temp2 = temp2[temp2 != 0]

        labels_v, conts_v = np.unique(temp2, return_counts=True)
        new_label = labels_v[np.argmax(conts_v)]

        #print(smaller_label, new_label)
        img_con[img_con == smaller_label] = new_label
        labels, conts = np.unique(img_con[img_con!=0], return_counts=True)

    return img_con