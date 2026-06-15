import streamlit as st

st.title("Corpus Callosum Parcellation Using TMG and SOM")

st.write("""
         This application implements the proposed data-driven Corpus Callosum (CC) parcellation method [[1]](#references),
         based on the Tensorial Morphological Gradient (TMG) map. The TMG [[2]](#references) highlights regions of higher
         dissimilarity between neighboring voxels by considering both intensity and directional information from Diffusion
         Tensor Imaging (DTI) data. The Self-Organizing Map (SOM) [[3]](#references) is then used to cluster voxels from the
         midsagittal slice of the CC into five regions using three-dimensional information captured by the TMG.
         """)

st.write("""
         More information about the method is provided below. To run the method on your data, go to the **Parcellation Setup**
         page in the sidebar.
         """)

#---------------------------------------

st.subheader("Tensorial Morphological Gradient of the Corpus Callosum")

st.write("""
         The method is based on the TMG map of the CC, computed with the Log-Euclidean distance (logE) [[4]](#references) as
         dissimilarity measure and using a 6-connected three-dimensional structuring element. For detailed information go to the
         **TMG Info** page in the sidebar.
         """)

st.write("""
         The TMG computation requires the DTI eigenvalues and eigenvectors for each subject. They must follow the DIPY convention,
         in which the eigenvectors are stored columnwise (i.e., the last array dimension represents the eigenvector).
         """)

st.write("""
         A binary CC mask must also be provided for each subject. The mask should include at least three CC slices:
         the midsagittal slice and its two neighboring slices. This requirement ensures the correct computation of
         the TMG using a 6-connected structuring element.
         """)

#---------------------------------------

st.subheader("Definition of the Midsagittal Slice")

st.write("""
         After computing the TMG map of the segmented CC, only the midsagittal slice is used for parcellation.
         This slice is identified using diffusion properties of the interhemispheric fissure, which typically contains
         large cerebrospinal fluid regions (low FA values) adjacent to white matter structures such as the CC (high FA values).
         """)
         
st.write("""
         Therefore, after discarding slices in which the cross-sectional area of the brain falls below a certain minimum
         (extremities slices), the midsagittal slice is defined as the one with the lowest average FA [[5]](#references).
         This process requires the FA map of each subject.
         """)

#---------------------------------------

st.subheader("Corpus Callosum Parcellation with Self-Organizing Maps")

st.markdown("""
            The SOM was implemented with the python package [sklearn-som](https://github.com/rileypsmith/sklearn-som),
            configured with a $\small 5 \\times 1$ grid and trained for $\small 10$ epochs. The TMG values in the midsagittal
            section of the CC are used as input to the SOM, together with their spatial coordinates. By applying the SOM algorithm,
            each voxel is assigned to one SOM unit, defining five classes.
            """)

st.write("""
         If this division generates more than five connected components, an additional step is performed: the smaller component
         is identified and its class is changed to the most frequent class occurring in its neighborhood. This step is repeated
         until there are only five components left.
         """)

st.write("""
         Since the weight vectors of the SOM are randomly initialized, the results may vary for each run of the method. To
         minimize this variability and generate more consistent parcellations, the procedure is repeated 100 times for each
         subject. Thus, the final parcellation is defined by majority of votes from each SOM execution.
         """)

#---------------------------------------

st.subheader("References")

st.markdown("""
            [1] Santana, C., Abreu, T., Rodrigues, J., Julio, P., Appenzeller, S., Rittner, L. (2023). DTI-based Corpus
            Callosum parcellation using the Tensorial Morphological Gradient and Self-Organizing Maps In: _2023 19th
            International Symposium on Medical Information Processing and Analysis (SIPAIM)_, doi:
            [10.1109/SIPAIM56729.2023.10373443](https://doi.org/10.1109/SIPAIM56729.2023.10373443).
            """)

st.markdown("""
            [2] Rittner, L., Campbell, J. S., Freitas, P. F., Appenzeller, S., Bruce Pike, G., Lotufo, R. A. (2013).
            Analysis of scalar maps for the segmentation of the corpus callosum in diffusion tensor fields. _Journal
            of mathematical imaging and vision_, doi: [10.1007/s10851-012-0377-4](https://doi.org/10.1007/s10851-012-0377-4).
            """)

st.markdown("""
            [3] Kohonen, T. (1990). The self-organizing map. _Proceedings of the IEEE_, doi:
            [10.1109/5.58325](https://doi.org/10.1109/5.58325).
            """)

st.markdown("""
            [4] Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2006). Log‐Euclidean metrics for fast and simple calculus
            on diffusion tensors. _Magnetic Resonance in Medicine_, doi: [10.1002/mrm.20965](https://doi.org/10.1002/mrm.20965).
            """)

st.markdown("""
            [5] Freitas, P., Rittner, L., Appenzeller, S., Lapa, A., Lotufo, R. (2012). Watershed-based segmentation of the
            corpus callosum in diffusion MRI. In: _Medical Imaging 2012: Image Processing_, doi:
            [10.1117/12.911619](https://doi.org/10.1117/12.911619).
            """)
