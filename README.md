# TMG-based Corpus Callosum Parcellation

This repository contains a Streamlit demo of the corpus callosum parcellation method proposed in:

> Santana, C., Abreu, T., Rodrigues, J., Julio, P., Appenzeller, S., Rittner, L.  
> DTI-based Corpus Callosum parcellation using the Tensorial Morphological Gradient (TMG) and Self-Organizing Maps (SOM).  
> In: _2023 19th International Symposium on Medical Information Processing and Analysis (SIPAIM)_.  
> doi: [10.1109/SIPAIM56729.2023.10373443](https://doi.org/10.1109/SIPAIM56729.2023.10373443)

To access the online demo, visit:

https://tmg-cc-parcellation.streamlit.app/

## TMG Standalone Usage

The following describes the TMG in detail and provides instructions for computing it independently of the parcellation pipeline.

### Background

The TMG [[ref](https://doi.org/10.1109/SIBGRAPI.2008.17)] uses the dissimilarity between neighboring voxels to summarize the main tensorial information into a scalar map. It adapts the idea of a morphological gradient to tensorial images and was proposed to facilitate segmentation and parcellation tasks using Diffusion Tensor Imaging (DTI) data.

For each voxel $v$ in a tensorial image $f$, the dissimilarity measure ($d_n$) between any pair of tensors ($T_i,T_j$) in its neighborhood (defined by the structuring element $B$) is calculated. The computed gradient will be the **maximum dissimilarity** among all pairwise dissimilarities:

$$
\nabla_B^T(f)(v) = \bigvee_{i,j \in B_v} d_n(T_i, T_j)
$$

There are several dissimilarity measures proposed for DTI applications and explored through the TMG [[ref](https://doi.org/10.1007/s10851-012-0377-4)]. This implementation supports the Dot Product (prod), Frobenius Norm (frob), Log-Euclidean distance (logE), and J-Divergence (Jdiv). We highlight the Frobenius Norm and the Log-Euclidean distance as the most promising metrics:

- Early TMG studies obtained the best results using the Frobenius Norm, which was attributed to its linear response to anisotropy and trace differences. Applications included [segmentation of the corpus callosum, the ventricles, and the cortico-spinal tract](https://ieeexplore.ieee.org/document/5395235) and [segmentation of thalamic nuclei](https://ieeexplore.ieee.org/abstract/document/5490203).
- The Log-Euclidean distance is closely related to the Frobenius Norm but uses the logarithm of the tensors. TMG results using both metrics look very similar but differ in scale, with the Log-Euclidean distance yielding larger values and higher contrast. It was applied to the [parcellation of the corpus callosum](https://ieeexplore.ieee.org/abstract/document/10373443).

The Dot Product only compares the principal eigenvectors of the tensors and might be too simple for most applications. In our experience, TMG results using the J-Divergence are difficult to handle when using real data, with some voxels showing extremely high values. For more details on the metrics, check [TMG Metrics Info and Examples on Synthetic Data.ipynb](<examples_TMG/TMG Metrics Info and Examples on Synthetic Data.ipynb>).

The choice of the structuring element (SE) generally depends on the information required (2D or 3D) and the size of the studied structure (smaller structuring elements tend to generate thinner, more detailed borders). This implementation provides 2-, 4-, 6-, and 8-connected SEs. Only the 6-connected SE is 3D. For 2D SEs, the desired orientation must be specified. To visualize different SE options, use [TMG Neighborhood Visualization.ipynb](<examples_TMG/TMG Neighborhood Visualization.ipynb>) (should be run locally and requires [DIPY and FURY](https://docs.dipy.org/stable/user_guide/installation)). Also, [TMGSE.py](<lib_TMG/TMGSE.py>) defines the SEs and provides a simple visualization.

### Instructions

The TMG computation uses DTI eigenvalues and eigenvectors. They **must follow the DIPY convention**, in which the eigenvectors are stored columnwise (the last dimension of the array defines the eigenvector). The main TMG function, including details on its inputs, is available in [TMG.py](<lib_TMG/TMG.py>). Examples of TMG computation using synthetic data can be found in [TMG Metrics Info and Examples on Synthetic Data.ipynb](<examples_TMG/TMG Metrics Info and Examples on Synthetic Data.ipynb>).