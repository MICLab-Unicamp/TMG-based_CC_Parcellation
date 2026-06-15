import numpy as np
from dipy.viz import window, actor, ui, read_viz_icons
from dipy.data import get_sphere
import dipy.reconst.dti as dipy_dti
from tqdm import tqdm

def t_RGB(evc, shape, fa):
    if fa is None:
        RGB = dipy_dti.color_fa(np.ones(shape), evc)
    else:
        RGB = dipy_dti.color_fa(fa, evc)
    return RGB

#Visualizar fatia única (tensores)
def tensorSlice(evals, evecs, axis, position, s = 'repulsion724', bg = (0.5, 0.5, 0.5), scale = 0.5, fa = None, norm = False):
    # axis -> define em relação a qual eixo será a fatia
    # position -> define a fatia a ser visualizada
    # s -> define a esfera
    # bg -> define a cor do background
    # scale -> define o fator de escala dos tensores
    # fa -> os valores de fa correspondentes devem ser passados caso se deseje considerá-los na coloração
    # norm -> define se os tensores gerados serão ou não normalizados

    #Para garantir que os autovalores e vetores originais não serão modificados
    eigvals = evals.copy()
    eigvecs = evecs.copy()
    shape = eigvals.shape[0:3]
    sphere = get_sphere(name=s)
    scene = window.Scene()
    scene.background(bg)
    
    scene.clear()
            
    cfa = t_RGB(eigvecs, shape, fa)

    slice_actor = actor.tensor_slicer(eigvals, eigvecs, scalar_colors=cfa, sphere=sphere, scale=scale, norm=norm)

    if axis == 'x':
        slice_actor.display_extent(position, position, 0, shape[1]-1, 0, shape[2]-1)
    elif axis == 'y':
        slice_actor.display_extent(0, shape[0]-1, position, position, 0, shape[2]-1)
    elif axis == 'z':
        slice_actor.display_extent(0, shape[0]-1, 0, shape[1]-1, position, position)

    scene.add(slice_actor)

    show_m = window.ShowManager(scene=scene, size=(1200, 900))
    show_m.initialize()

    scene.zoom(1.5)
    scene.reset_clipping_range()

    show_m.render()
    show_m.start()
    
    return

#Visualização dinâmica de fatias (tensores)    
def dynamicTensorSlice(evals, evecs, s = 'repulsion724', bg = (0.5, 0.5, 0.5), scale = 0.5, fa=None, norm=False):    
    # s -> define a esfera
    # bg -> define a cor do background
    # scale -> define o fator de escala dos tensores
    # fa -> os valores de fa correspondentes devem ser passados caso se deseje considerá-los na coloração
    # norm -> define se os tensores gerados serão ou não normalizados
    
    #Para garantir que os autovalores e vetores originais não serão modificados
    eigvals = evals.copy()
    eigvecs = evecs.copy()

    shape = eigvals.shape[0:3]
    sphere = get_sphere(name=s)
    scene = window.Scene()
    scene.background(bg)
    
    if fa is None:
        RGB = dipy_dti.color_fa(np.ones(shape), eigvecs)
    else:
        RGB = dipy_dti.color_fa(fa, eigvecs)
        
    cfa = RGB
    
    scene.clear()    

    #The default behavior of the tensor_slicer is to show the middle slice of the last dimension of the data (z)
    slice_actor_z = actor.tensor_slicer(eigvals, eigvecs, scalar_colors=cfa, sphere=sphere, scale=scale, norm=norm)

    #Creating the middle x slice
    x_midpoint = int(np.round(shape[0]/2))
    slice_actor_x = actor.tensor_slicer(eigvals, eigvecs, scalar_colors=cfa, sphere=sphere, scale=scale, norm=norm)
    slice_actor_x.display_extent(x_midpoint, x_midpoint, 0, shape[1]-1, 0, shape[2]-1)

    #Creating the middle y slice
    slice_actor_y = actor.tensor_slicer(eigvals, eigvecs, scalar_colors=cfa, sphere=sphere, scale=scale, norm=norm)
    y_midpoint = int(np.round(shape[1]/2))
    slice_actor_y.display_extent(0, shape[0]-1, y_midpoint, y_midpoint, 0, shape[2]-1)

    #Connect the actors with the Scene
    scene.add(slice_actor_z)
    scene.add(slice_actor_x)
    scene.add(slice_actor_y)

    show_m = window.ShowManager(scene=scene, size=(1200, 900))
    show_m.initialize()

    #Creating sliders to move the slices
    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)
    
    #Writing callbacks for the sliders and registering them
    def change_slice_z(slider):
        z = int(np.round(slider.value))
        slice_actor_z.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        slice_actor_x.display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        slice_actor_y.display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)
    
    line_slider_z.on_change = change_slice_z
    line_slider_x.on_change = change_slice_x
    line_slider_y.on_change = change_slice_y
    
    #Creating text labels to identify the sliders
    def build_label(text):
        label = ui.TextBlock2D()
        label.message = text
        label.font_size = 18
        label.font_family = 'Arial'
        label.justification = 'left'
        label.bold = False
        label.italic = False
        label.shadow = False
        label.background_color = (0, 0, 0)
        label.color = (1, 1, 1)

        return label

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")

    #Creating a panel to contain the sliders and labels
    panel = ui.Panel2D(size=(300, 200),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")
    panel.center = (1030, 120)

    panel.add_element(line_slider_label_x, (0.1, 0.75))
    panel.add_element(line_slider_x, (0.38, 0.75))
    panel.add_element(line_slider_label_y, (0.1, 0.55))
    panel.add_element(line_slider_y, (0.38, 0.55))
    panel.add_element(line_slider_label_z, (0.1, 0.35))
    panel.add_element(line_slider_z, (0.38, 0.35))

    scene.add(panel)

    #Update the position of the panel using its re_align method every time the window size change
    global size
    size = scene.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel.re_align(size_change)
    
    scene.zoom(1.5)
    scene.reset_clipping_range()

    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()
    
    return

#Visualizar volume completo
def volumeTensor(evals, evecs, s = 'repulsion724', bg = (0.5, 0.5, 0.5), scale = 0.5, fa=None, norm=False, opacity = 1):
    # s -> define a esfera
    # bg -> define a cor do background
    # scale -> define o fator de escala dos tensores
    # fa -> os valores de fa correspondentes devem ser passados caso se deseje considerá-los na coloração
    # norm -> define se os tensores gerados serão ou não normalizados

    #Para garantir que os autovalores e vetores originais não serão modificados
    eigvals = evals.copy()
    eigvecs = evecs.copy()

    shape = eigvals.shape[0:3]
    sphere = get_sphere(name=s)
    scene = window.Scene()
    scene.background(bg)
    
    if fa is None:
        RGB = dipy_dti.color_fa(np.ones(shape), eigvecs)
    else:
        RGB = dipy_dti.color_fa(fa, eigvecs)
        
    cfa = RGB

    scene.clear()

    #Creating all the z slices
    for i in tqdm(range(shape[0])):
        slice_actor_z = actor.tensor_slicer(eigvals, eigvecs, scalar_colors=cfa, sphere=sphere, scale=scale, norm=norm, opacity=opacity)
        slice_actor_z.display_extent(0, shape[0]-1, 0, shape[1]-1, i, i)
        scene.add(slice_actor_z)

    show_m = window.ShowManager(scene=scene, size=(1200, 900))
    show_m.initialize()

    scene.zoom(1.5)
    scene.reset_clipping_range()

    show_m.render()
    show_m.start()
    
    return