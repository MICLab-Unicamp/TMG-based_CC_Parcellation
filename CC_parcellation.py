import streamlit as st
import os
from lib_TMG.TMG import TMG
from lib_SOM.SOM_funcs_streamlit import SOM_parc
from lib_TMG.TMGUtil import cut_imgs_mask, vis_imgs_2D
from dipy.io.image import load_nifti, save_nifti
import numpy as np
import zipfile
import tempfile

st.title('Corpus Callosum parcellation using TMG and SOM')

st.write('The proposed Corpus Callosum (CC) parcellation is a data-driven method based on the Tensorial Morphological Gradient (TMG) map. The TMG [[1]](#references) highlights regions of higher dissimilarity\
          between neighbor voxels taking into account not only intensity but also directional information from Diffusion Tensor Imaging (DTI) data. Then, the Self-Organizing Map (SOM) [2] is\
          used for clustering the voxels of the midsagittal slice of the CC into five regions, taking into account three-dimensional information captured by the TMG.')

#---------------------------------------

st.subheader('Tensorial Morphological Gradient of the Corpus Callosum')

st.write('The method is based on the TMG map of the CC, computed with the Log-Euclidean distance (logE) [3] as dissimilarity measure and using a 6-connected three-dimensional structuring element.\
          For detailed information, see page _TMG info_.')

st.write('The TMG calculation requires the DTI eigenvalues and eigenvectors of each subject. Note that those must follow the DIPY convention, in which the eigenvectors are stored columnwise\
         (the last dimension of the array defines the eigenvector).')

st.write('Also, as the starting point for the parcellation method, a binary CC mask must be provided for each subject. It should include at least three CC slices: the midsagittal slice and the two\
         slices neighboring it. This is necessary for the correct calculation of the TMG using a 6-connected structuring element.')

#---------------------------------------

st.subheader('Definition of the Midsagittal Slice')

st.write('After obtaining the TMG of the segmented CC, only its midsagittal slice is considered for the parcellation. It is identified by the diffusion properties of the\
          inter-hemispheric fissure of the brain, consisting of large areas of cerebrospinal fluid (low FA values) and white matter structures such as the CC (high FA values).')
         
st.write('Therefore, after discarding slices in which the cross-sectional area of the brain falls below a certain minimum (extremities slices), the midsagittal slice is defined\
          as the one with the lowest average FA [4]. This process requires the FA map of each subject.')

#---------------------------------------

st.subheader('Corpus Callosum Parcellation with Self-Organizing Maps')

st.markdown("""The SOM was implemented with the python package [sklearn-som](https://github.com/rileypsmith/sklearn-som), configured to have a grid of $5$x$1$ and $10$ epochs.\
            The TMG values in the midsagittal section of the CC are used as input to the SOM, together with their spatial coordinates. By applying the SOM algorithm,\
            each voxel is assigned to one of the map units, defining five classes.""")

st.write('If this division generates more than five connected components, an additional step is performed: the smaller component is identified and its class is changed to the most\
          frequent class occurring in its neighborhood. This step is repeated until there are only five components left.')

st.write('Since the weight vectors of the SOM are randomly initialized, the results may vary for each run of the method. To minimize this variability and generate more consistent\
          parcellations, the procedure is repeated 100 times for each subject. Thus, the final parcellation is defined by majority of votes from each SOM execution.')

#---------------------------------------

st.subheader('References')

st.markdown("""[1] Rittner, L., Campbell, J. S., Freitas, P. F., Appenzeller, S., Bruce Pike, G., Lotufo, R. A. (2013). Analysis of scalar maps for the segmentation of the corpus callosum \
            in diffusion tensor fields. _Journal of mathematical imaging and vision_, 45, 214-226, doi: [10.1007/s10851-012-0377-4](https://doi.org/10.1007/s10851-012-0377-4).""")

st.markdown("""[2] Kohonen, T. (1990). The self-organizing map. _Proceedings of the IEEE_, 78(9), 1464-1480, doi: [10.1109/5.58325](https://doi.org/10.1109/5.58325).""")

st.markdown("""[3] Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2006). Log‐Euclidean metrics for fast and simple calculus on diffusion tensors. _Magnetic Resonance in Medicine_, \
            56(2), 411-421, doi: [10.1002/mrm.20965](https://doi.org/10.1002/mrm.20965).""")

st.markdown("""[4] Freitas, P., Rittner, L., Appenzeller, S., Lapa, A., Lotufo, R. (2012). Watershed-based segmentation of the corpus callosum in diffusion MRI. In _Medical Imaging 2012:_ \
            _Image Processing_ (Vol. 8314, pp. 879-885). SPIE, doi: [10.1117/12.911619](https://doi.org/10.1117/12.911619).""")

#---------------------------------------

st.subheader('Configuration')

#----------

st.write('**INPUT**')

st.write('Each subject data should be uploaded as a .zip file containing all the required files (eigenvalues, eigenvectors, CC mask, and FA map) in .nii or .nii.gz format.\
         Note that the uploader (bellow) accepts multiple files/subjects.')

uploaded_files = st.file_uploader("Select file(s):", type='zip', accept_multiple_files=True)

expander_fnames = st.expander('Expand to specify the names of the required files', expanded=False)
evals_file_name = expander_fnames.text_input('Filename of the eigenvalues:', 'evals.nii.gz')
evecs_file_name = expander_fnames.text_input('Filename of the eigenvectors:', 'evecs.nii.gz')
mask_file_name = expander_fnames.text_input('Filename of the mask:', 'CC_mask_CNN.nii.gz')
fa_file_name = expander_fnames.text_input('Filename of the FA:', 'FA.nii.gz')

#----------

st.write('**ADVANCED CONFIGURATION**')

st.write('It is not recommended to change the configurations bellow, since they correspond to the proposed method by default. However, it is possible explore\
         other configurations for the TMG and SOM.')

expander_adv_tmg = st.expander("Advanced TMG configuration", expanded=False)
expander_adv_tmg.write('For detailed information about the TMG parameters, see page _TMG info_.')
metric = expander_adv_tmg.selectbox('Select dissimilarity measure:', ['prod', 'frob', 'Jdiv', 'logE'], 3)
nbh = expander_adv_tmg.selectbox('Select structuring element:', [2, 4, 6, 8], 2)
dict_ori = {2: ['x', 'y', 'z'], 4: ['xy', 'xz', 'yz'], 6: [''], 8: ['xy', 'xz', 'yz']}
nbh_ori = expander_adv_tmg.selectbox('Select orientation of the structuring element:', dict_ori[nbh])

expander_adv_som = st.expander("Advanced SOM configuration", expanded=False)
n_labels = expander_adv_som.number_input('Number of labels:', value=5, min_value=1)
epochs = expander_adv_som.number_input('Number of epochs:', value=10, min_value=1)
n_rep = expander_adv_som.number_input('Number of repetitions:', value=100, min_value=1)

#----------

st.write('**OUTPUT**')

st.write('By default, the output will be a .zip file with the parcellation mask ("CC_parc_SOM.nii.gz") of each subject in its respective folder. Other output options can be selected bellow.')

expander_outputs = st.expander('Output options', expanded=False)
expander_outputs.checkbox(f'Save TMG map', key="save_tmg")
out_file_suffix = expander_outputs.text_input("TMG filename suffix:", "_CC_CNN", disabled=not st.session_state.save_tmg, key = 'out_file_suffix')
expander_outputs.checkbox(f'Save each region as a separated binary mask file', key="save_parcels")
expander_outputs.checkbox(f'Save file with the "certainty" of the parcellation (the proportion of times the SOM execution indicated the final class attributed to each voxel)', key="save_certain")

#----------

st.markdown("""---""")

st.write('**CURRENT CONFIGURATION**')

st.write(f'**TMG:** measure {metric}, structuring element {nbh}{nbh_ori}.', )

st.write(f'**SOM:** {n_labels} labels, {epochs} ephocs, {n_rep} repetitions.', )

saves = ''
if st.session_state.save_tmg: saves+=', TMG'
if st.session_state.save_parcels: saves+=', individual parcels'
if st.session_state.save_certain: saves+=', parcellation certainty'

st.write(f'**Output:** full parcellation' + saves + '.')

if len(uploaded_files) == 0:
    st.write('**:red[Upload the file(s) to allow the parcellation!]**')
else:
    st.write('**If the configuration is correct, click in the button bellow to perform the parcellation. The last four parcellation results will be presented (on-the-fly) for visualization.\
             After completion, a new button will appear to download the results.**')

st.markdown("""---""")

#---------------------------------------

if 'output' not in st.session_state:
    st.session_state.output = None

if 'output_vis' not in st.session_state:
    st.session_state.output_vis = None

# Add a placeholder
latest_iteration = st.empty()
if st.session_state.output is not None:
    latest_iteration.write('**:green[Completed!]**')
    bar = st.progress(100)
    st.download_button(label="Click to download the results", data=st.session_state.output, file_name= 'output.zip', mime = 'zip', key="download_zip", type = 'primary')
else:
    bar = st.progress(0)

vis_info = st.empty()
fig_parcs = st.empty()
if st.session_state.output_vis is not None:
    vis_info.write('**Last parcellation results obtained:**')
    fig_parcs = st.pyplot(st.session_state.output_vis)

def click_button():

    temp_dir_out = tempfile.TemporaryDirectory()
    output_zip_file = os.path.join(temp_dir_out.name, 'output.zip')
    size = len(uploaded_files)
    imgs_vis = []
    titles = []

    for i,uploaded_file in enumerate(uploaded_files):

        latest_iteration.write(f'**:blue[Computing subject {i+1} of {size}, file {uploaded_file.name}]**')
        bar.progress((i)*(1/size))

        # Create a temporary directory to store extracted files
        temp_dir = tempfile.TemporaryDirectory()
        id = os.path.splitext(uploaded_file.name)[0]

        # Read the uploaded .zip file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # Extract the files to the temporary directory
            zip_ref.extractall(temp_dir.name)
            # Gettin files
            evals,affine = load_nifti(os.path.join(temp_dir.name, evals_file_name))
            evecs,_ = load_nifti(os.path.join(temp_dir.name, evecs_file_name))
            mask,_ = load_nifti(os.path.join(temp_dir.name, mask_file_name))
            fa,_ = load_nifti(os.path.join(temp_dir.name, fa_file_name))

        # Clean up the temporary directory when the app is done
        temp_dir.cleanup()

        # Calculate TMG
        tmg = TMG(evals,evecs,metric,nbh,nbh_ori,mask=mask)

        if not os.path.isdir(os.path.join(temp_dir_out.name, id)):
            os.makedirs(os.path.join(temp_dir_out.name, id))

        #Save TMG
        if st.session_state.save_tmg:
            save_nifti(os.path.join(temp_dir_out.name, id, f'TMG_{metric}_{nbh}{nbh_ori}{out_file_suffix}.nii.gz'), np.float32(tmg), affine)

        # Apply SOM
        final_parc, certainty = SOM_parc(tmg, mask, evals, fa, n_labels=n_labels, epochs=epochs, n_rep=n_rep)
        #Correcting labels
        final_parc[final_parc != 0] = final_parc[final_parc != 0]*(-1) + 6

        # Save parcellation
        save_nifti(os.path.join(temp_dir_out.name, id, 'CC_parc_SOM.nii.gz'), final_parc, affine)

        # Save parcels
        if st.session_state.save_parcels:
            # Generating masks
            region_I = final_parc.copy()
            region_I[region_I != 1] = 0
            region_II = final_parc.copy()
            region_II[region_II != 2] = 0
            region_II[region_II == 2] = 1
            region_III = final_parc.copy()
            region_III[region_III != 3] = 0
            region_III[region_III == 3] = 1
            region_IV = final_parc.copy()
            region_IV[region_IV != 4] = 0
            region_IV[region_IV == 4] = 1
            region_V = final_parc.copy()
            region_V[region_V != 5] = 0
            region_V[region_V == 5] = 1
            # Saving masks
            save_nifti(os.path.join(temp_dir_out.name, id, 'CC_parc_SOM_I.nii.gz'), region_I, affine)
            save_nifti(os.path.join(temp_dir_out.name, id, 'CC_parc_SOM_II.nii.gz'), region_II, affine)
            save_nifti(os.path.join(temp_dir_out.name, id, 'CC_parc_SOM_III.nii.gz'), region_III, affine)
            save_nifti(os.path.join(temp_dir_out.name, id, 'CC_parc_SOM_IV.nii.gz'), region_IV, affine)
            save_nifti(os.path.join(temp_dir_out.name, id, 'CC_parc_SOM_V.nii.gz'), region_V, affine)

        # Save certainty
        if st.session_state.save_certain:
            save_nifti(os.path.join(temp_dir_out.name, id, 'Parcellation_certainty.nii.gz'), certainty, affine)

        # Visualization of the results
        parc_cut,_,_ = cut_imgs_mask(final_parc, final_parc, bin=False)

        parc_rgb = np.ones(np.append(parc_cut.shape,3))*0.9
        parc_rgb[parc_cut == 1] = [102/255,194/255,165/255]
        parc_rgb[parc_cut == 2] = [252/255,141/255,98/255]
        parc_rgb[parc_cut == 3] = [141/255,160/255,203/255]
        parc_rgb[parc_cut == 4] = [231/255,138/255,195/255]
        parc_rgb[parc_cut == 5] = [166/255,216/255,84/255]

        imgs_vis.insert(0,parc_rgb)
        titles.insert(0,id)

        if i > 3:
            imgs_vis.pop(-1)
            titles.pop(-1)

        vis_info.write('**Last parcellation results obtained:**')
        fig_parcs.pyplot(vis_imgs_2D(imgs_vis, 'yz', 0, 1, titles, return_fig=True, f_m_s_c=10, f_m_s_r=4))

    bar.progress((i)*(1/size))

    # Create a .zip file and add the files from the directory
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir_out.name):
            for file in files:
                if not file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, temp_dir_out.name))

    # Provide a download button to download the generated .zip file
    with open(output_zip_file, "rb") as f:
        zip_file_bytes = f.read()
    st.session_state.output=zip_file_bytes

    st.session_state.output_vis = vis_imgs_2D(imgs_vis, 'yz', 0, 1, titles, return_fig=True, f_m_s_c=10, f_m_s_r=4)

    temp_dir_out.cleanup()

if len(uploaded_files) > 0:
    st.button('**Parcelate CC**', on_click=click_button)