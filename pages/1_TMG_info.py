import streamlit as st

st.title('Tensorial Morphological Gradient (TMG)')

st.write('The TMG [[1]](#references) uses the dissimilarity between neighboring voxels to summarize the main tensorial information into a scalar map. It adapts the idea of a morphological\
          gradient to tensorial images and was proposed to facilitate segmentation and parcellation tasks using Diffusion Tensor Imaging (DTI) data.')

st.write('For each voxel $v$ in a tensorial image $f$, the dissimilarity measure $(d_n)$ between any pair of tensors $(T_i,T_j)$ in its neighborhood is calculated. The neighborhood is defined\
          by a structuring element $B$. The computed gradient will be the maximum dissimilarity among all pairwise dissimilarities:')

st.latex(r'''\nabla_B^T(f)(v) = \bigvee_{i,j \in B_v} d_n(T_i, T_j)''')

st.write('There are several dissimilarity measures proposed for DTI applications and explored through the TMG [[1]](#references). The choice of the structuring element generally depends on the\
          information required (bi- or three-dimensional) and the size of the studied structure - smaller structuring elements tend to generate thinner, more detailed borders.')

#---------------------------------------

st.header('Dissimilarity measures')

#-----------------

st.subheader('Dot Product (prod)')

st.write('The Dot Product [[2]](#references) only takes in consideration the principal eigenvectors of the tensors. The result varies between 0 and 1 according to the angle between the\
          vectors: equals to one if the vectors are parallel and equals to zero if the vectors are orthogonal.')

st.latex(r'''d_{prod}(T_i, T_j) = |e_{1i} \cdot e_{1j}|''')

st.write('Since the Dot Product is a similarity measure, the TMG calculates the minimal similarity instead of the maximal dissimilarity. The negative of the resulting image\
          will be the output of the TMG (maintaining the same pattern as for the other measures).')

#-----------------

st.subheader('Frobenius Norm (frob)')

st.write('The Frobenius Norm [[3]](#references) is an Euclidean distance measure between the tensors, as follows:')

st.latex(r'''d_{frob}(T_i,T_j) = \sqrt{trace((T_i - T_j)^2)}''')

st.write('This dissimilarity measure is not invariant to affine transformations. However, it presents a linear response to differences in anisotropy and trace.')

#-----------------

st.subheader('J-Divergence (Jdiv)')

st.write('The J-divergence [[4]](#references) is a tensorial "distance" measure, based on concepts of theory information. The measure is not a true distance, since it violates the triangle\
          inequality. However, it is a computationally efficient approximation of the Rao\'s distance and is invariant to affine transformations.')

st.latex(r'''d_{Jdiv}(T_i,T_j) = \frac{1}{2} \sqrt{trace(T_i^{-1}T_j + T_j^{-1}T_i) - 2n}''')

st.markdown("""with $n$ being the size of the square matrix $T$.""")

#-----------------

st.subheader('Log-Euclidean distance (logE)')

st.write('The logE [[5]](#references) is an Euclidean distance measure between the logarithm of the tensors, close related to the Frobenius Norm.')

st.latex(r'''d_{logE}(T_i, T_j) = \sqrt{trace((\log(T_i)-\log(T_j))^2)}''')

st.write('From the equation above, it is possible to see that symmetric matrices with null or negative eigenvalues are at an infinite distance from any tensor.\
          To overcome this problem, $\log(T_i)$ was replaced with $E_i \cdot \log(100\lambda_i+1)I\cdot E_i^T$, where $E$ corresponds to the matrix of eigenvectors,\
          $\lambda$ corresponds to the eigenvalues, and $I$ is the identity matrix.')

#---------------------------------------

st.subheader('References')

st.markdown("""[1] Rittner, L., Campbell, J. S., Freitas, P. F., Appenzeller, S., Bruce Pike, G., Lotufo, R. A. (2013). Analysis of scalar maps for the segmentation of the corpus callosum \
            in diffusion tensor fields. _Journal of mathematical imaging and vision_, doi: [10.1007/s10851-012-0377-4](https://doi.org/10.1007/s10851-012-0377-4).""")

st.markdown("""[2] Ziyan, U., Tuch, D., Westin, C. F. (2006). Segmentation of thalamic nuclei from DTI using spectral clustering. In _Medical Image Computing and Computer-Assisted \
            Intervention–MICCAI 2006_, doi: [10.1007/11866763_99](https://doi.org/10.1007/11866763_99).""")

st.markdown("""[3] Alexander, D. C., Gee, J. C., Bajcsy, R. (1999). Similarity Measures for Matching Diffusion Tensor Images. In _BMVC_.""")

st.markdown("""[4] Wang, Z., Vemuri, B. C. (2005). DTI segmentation using an information theoretic tensor dissimilarity measure. _IEEE transactions on medical imaging_, doi: \
            [10.1109/TMI.2005.854516](https://doi.org/10.1109/TMI.2005.854516).""")

st.markdown("""[5] Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2006). Log‐Euclidean metrics for fast and simple calculus on diffusion tensors. Magnetic Resonance in Medicine, doi: \
            [10.1002/mrm.20965](https://doi.org/10.1002/mrm.20965)""")