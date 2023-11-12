def TMG(eigvals,eigvects,similarity,neighborhood=6,nbh_info=None,mask=None):
    import numpy as np
    from lib_TMG import TMGMetrics as mtc
    from lib_TMG import TMGSE as se

    dict_nbh = {2: se.two_connected, 4: se.four_connected, 6: se.six_connected, 8: se.eight_connected}

    # Cópia das matrizes de entrada, para não alterá-las
    eigvalslin = eigvals.copy()
    eigvectslin = eigvects.copy()

    if mask is not None:
        # Cópia das matrizes de entrada, para não alterá-las
        mask_norm = mask.copy()
        # Garante que a máscara é binária
        mask_norm[mask_norm != 0] = 1

        # Obtém os índices dos voxels de interesse
        roi_ind = np.where(mask_norm == 1)
        # Pega os máximos e mínimos em cada dimensão (extremidades da máscara)
        min_x = np.min(roi_ind[0])
        max_x = np.max(roi_ind[0])
        min_y = np.min(roi_ind[1])
        max_y = np.max(roi_ind[1])
        min_z = np.min(roi_ind[2])
        max_z = np.max(roi_ind[2])

        # Bounding box em torno da máscara para evitar cálculos desnecessários
        eigvalslin = eigvalslin[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
        eigvectslin = eigvectslin[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
        masccut = np.int8(mask_norm[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1])

        # Zera os voxels de fundo para evitar cálculos desnecessários
        masklin = np.expand_dims(masccut, axis = -1)
        eigvalslin = eigvalslin*masklin

        masklin = np.expand_dims(masklin, axis = -1)
        eigvectslin = eigvectslin*masklin

        # Achata a matriz
        masklin.shape = np.prod(masklin.shape)

    shape = eigvectslin.shape

    # Transforma as 3 primeiras dimensões em uma só
    eigvalslin.shape = np.append(np.prod(eigvalslin.shape[0:3]),eigvalslin.shape[3])
    eigvectslin.shape = np.append(np.prod(eigvectslin.shape[0:3]),eigvectslin.shape[3::])
    
    # Calcula para cada voxel os índices dos voxels que compõem a sua vizinhança
    indices = dict_nbh[neighborhood](shape, nbh_info)

    # Seleciona
    eigvalsb = eigvalslin[indices,:]
    eigvectsb = eigvectslin[indices,:,:]

    if mask is not None:
        maskb = masklin[indices]
    else:
        maskb = None

    # Calcula as distâncias
    distance = mtc.tensorialSimilarityMeasures(eigvalsb,eigvectsb,similarity,neighborhood,mask=maskb)

    if mask is not None:
        # Multiplico a distância pela máscara para garantir que os voxels de fundo são todos zeros
        distance = distance*np.expand_dims(masccut, axis = -1)
    
    # Calcula o TMG propriamente dito
    if (similarity == 'prod'):
        # Para métricas de similaridade, calcula o mínimo entre as distâncias
        if mask is not None:
            img = np.zeros(mask.shape, dtype=distance.dtype)
            img[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1] = distance.min(axis=-1)
            # Calcula o negativo do prod (apenas no interior da máscara) para ficar no padrão das outras métricas
            img[mask_norm == 1] = img[mask_norm == 1].max()-img[mask_norm == 1]
        else:
            img = distance.min(axis=-1)
            img = img.max() - img
        
    else:
        # Para métricas de dissimilaridade, calcula o máximo entre as distâncias
        if mask is not None:
            img = np.zeros(mask.shape, dtype=distance.dtype)
            img[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1] = distance.max(axis=-1)
        else:
            img = distance.max(axis=-1)

    return img

def T_TMG(eigvals,eigvects,similarity,neighborhood=2,mask=None):
    import numpy as np

    if neighborhood == 2:
        nbh_infos = ['x', 'y', 'z']
    else:
        nbh_infos = ['yz', 'xz', 'xy']

    img = np.zeros(np.append(eigvects.shape[0:3],3), dtype='float32')

    for i, info in enumerate(nbh_infos):
        img_temp = TMG(eigvals,eigvects,similarity,neighborhood,info,mask)
        img[...,i] = img_temp

    return img
