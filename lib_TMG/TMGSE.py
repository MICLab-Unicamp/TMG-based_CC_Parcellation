
def two_connected(shape, info):
    import numpy as np

    fat = shape[2]
    lin = shape[1]
    col = shape[0]

    i,j,k = np.indices((col,lin,fat))

    indices = np.zeros(np.append(shape[0:3],3), dtype='uint32')

    indices[...,0] = lin*fat*i+fat*j+k

    if info == 'x':
        indices[...,1] = lin*fat*np.maximum(i-1,0)+fat*j+k
        indices[...,2] = lin*fat*np.minimum(i+1,col-1)+fat*j+k
    elif info == 'y':
        indices[...,1] = lin*fat*i+fat*np.maximum(j-1,0)+k
        indices[...,2] = lin*fat*i+fat*np.minimum(j+1,lin-1)+k
    elif info == 'z':
        indices[...,1] = lin*fat*i+fat*j+np.maximum(k-1,0)
        indices[...,2] = lin*fat*i+fat*j+np.minimum(k+1,fat-1)

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |    |  x |    |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |  z |    |      |  y |  * |  y |        |    |  z |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |    |  x |    |        |    |    |    |

    return indices


def four_connected(shape, info):
    import numpy as np

    fat = shape[2]
    lin = shape[1]
    col = shape[0]

    i,j,k = np.indices((col,lin,fat))

    indices = np.zeros(np.append(shape[0:3],5), dtype='uint32')

    indices[...,0] = lin*fat*i+fat*j+k

    if info == 'xy':
        indices[...,1] = lin*fat*i+fat*np.maximum(j-1,0)+k
        indices[...,2] = lin*fat*np.maximum(i-1,0)+fat*j+k
        indices[...,3] = lin*fat*np.minimum(i+1,col-1)+fat*j+k
        indices[...,4] = lin*fat*i+fat*np.minimum(j+1,lin-1)+k
    elif info == 'xz':
        indices[...,1] = lin*fat*i+fat*j+np.maximum(k-1,0)
        indices[...,2] = lin*fat*np.maximum(i-1,0)+fat*j+k
        indices[...,3] = lin*fat*np.minimum(i+1,col-1)+fat*j+k
        indices[...,4] = lin*fat*i+fat*j+np.minimum(k+1,fat-1)
    elif info == 'yz':
        indices[...,1] = lin*fat*i+fat*j+np.maximum(k-1,0)
        indices[...,2] = lin*fat*i+fat*np.maximum(j-1,0)+k
        indices[...,3] = lin*fat*i+fat*np.minimum(j+1,lin-1)+k
        indices[...,4] = lin*fat*i+fat*j+np.minimum(k+1,fat-1)


    # xy plane

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |    |  1 |    |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |  2 |  0 |  3 |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |    |  4 |    |        |    |    |    |


    # xz plane

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |    |    |    |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |  1 |    |      |  2 |  0 |  3 |        |    |  4 |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |    |    |    |        |    |    |    |


    # yz plane

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |    |  2 |    |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |  1 |    |      |    |  0 |    |        |    |  4 |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |    |  3 |    |        |    |    |    |

    return indices


def six_connected(shape, info):
    import numpy as np

    fat = shape[2]
    lin = shape[1]
    col = shape[0]

    i,j,k = np.indices((col,lin,fat))

    indices = np.zeros(np.append(shape[0:3],7), dtype='uint32')

    indices[...,0] = lin*fat*i+fat*j+np.maximum(k-1,0)
    indices[...,1] = lin*fat*i+fat*np.maximum(j-1,0)+k
    indices[...,2] = lin*fat*np.maximum(i-1,0)+fat*j+k
    indices[...,3] = lin*fat*i+fat*j+k
    indices[...,4] = lin*fat*np.minimum(i+1,col-1)+fat*j+k
    indices[...,5] = lin*fat*i+fat*np.minimum(j+1,lin-1)+k
    indices[...,6] = lin*fat*i+fat*j+np.minimum(k+1,fat-1)


    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |    |  1 |    |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |  0 |    |      |  2 |  3 |  4 |        |    |  6 |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |    |  5 |    |        |    |    |    |

    return indices


def eight_connected(shape, info):
    import numpy as np

    fat = shape[2]
    lin = shape[1]
    col = shape[0]

    i,j,k = np.indices((col,lin,fat))

    indices = np.zeros(np.append(shape[0:3],9), dtype='uint32')

    if info == 'xy':
        indices[...,0] = lin*fat*np.maximum(i-1,0)+fat*np.maximum(j-1,0)+k
        indices[...,1] = lin*fat*i+fat*np.maximum(j-1,0)+k
        indices[...,2] = lin*fat*np.minimum(i+1,col-1)+fat*np.maximum(j-1,0)+k
        indices[...,3] = lin*fat*np.maximum(i-1,0)+fat*j+k
        indices[...,4] = lin*fat*i+fat*j+k
        indices[...,5] = lin*fat*np.minimum(i+1,col-1)+fat*j+k
        indices[...,6] = lin*fat*np.maximum(i-1,0)+fat*np.minimum(j+1,lin-1)+k
        indices[...,7] = lin*fat*i+fat*np.minimum(j+1,lin-1)+k
        indices[...,8] = lin*fat*np.minimum(i+1,col-1)+fat*np.minimum(j+1,lin-1)+k
    elif info == 'xz':
        indices[...,0] = lin*fat*np.maximum(i-1,0)+fat*j+np.maximum(k-1,0)
        indices[...,1] = lin*fat*i+fat*j+np.maximum(k-1,0)
        indices[...,2] = lin*fat*np.minimum(i+1,col-1)+fat*j+np.maximum(k-1,0)
        indices[...,3] = lin*fat*np.maximum(i-1,0)+fat*j+k
        indices[...,4] = lin*fat*i+fat*j+k
        indices[...,5] = lin*fat*np.minimum(i+1,col-1)+fat*j+k
        indices[...,6] = lin*fat*np.maximum(i-1,0)+fat*j+np.minimum(k+1,fat-1)
        indices[...,7] = lin*fat*i+fat*j+np.minimum(k+1,fat-1)
        indices[...,8] = lin*fat*np.minimum(i+1,col-1)+fat*j+np.minimum(k+1,fat-1)
    elif info == 'yz':
        indices[...,0] = lin*fat*i+fat*np.maximum(j-1,0)+np.maximum(k-1,0)
        indices[...,1] = lin*fat*i+fat*j+np.maximum(k-1,0)
        indices[...,2] = lin*fat*i+fat*np.minimum(j+1,lin-1)+np.maximum(k-1,0)
        indices[...,3] = lin*fat*i+fat*np.maximum(j-1,0)+k
        indices[...,4] = lin*fat*i+fat*j+k
        indices[...,5] = lin*fat*i+fat*np.minimum(j+1,lin-1)+k
        indices[...,6] = lin*fat*i+fat*np.maximum(j-1,0)+np.minimum(k+1,fat-1)
        indices[...,7] = lin*fat*i+fat*j+np.minimum(k+1,fat-1)
        indices[...,8] = lin*fat*i+fat*np.minimum(j+1,lin-1)+np.minimum(k+1,fat-1)


    # xy plane

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |  0 |  1 |  2 |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |  3 |  4 |  5 |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |  6 |  7 |  8 |        |    |    |    |


    # xz plane

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |    |    |      |    |    |    |        |    |    |    |
    # ----------------      ----------------        ----------------
    # |  0 |  1 |  2 |      |  3 |  4 |  5 |        |  6 |  7 |  8 |
    # ----------------      ----------------        ----------------
    # |    |    |    |      |    |    |    |        |    |    |    |


    # yz plane

    #Fatia inferior         Fatia do meio           Fatia superior

    # |    |  0 |    |      |    |  3 |    |        |    |  6 |    |
    # ----------------      ----------------        ----------------
    # |    |  1 |    |      |    |  4 |    |        |    |  7 |    |
    # ----------------      ----------------        ----------------
    # |    |  2 |    |      |    |  5 |    |        |    |  8 |    |

    return indices