import streamlit as st

st.title('Tensorial Morphological Gradient (TMG)')

st.write('The TMG uses the dissimilarity between neighboring voxels to summarize the main tensorial information into a scalar map. It was proposed to facilitate segmentation\
          and parcellation tasks using DTI data.')

st.write('For each voxel $v$ in a tensorial image $f$, the dissimilarity measure $(d_n)$ between any pair of tensors $(T_i,T_j)$ in its neighborhood (defined by the structuring\
          element $B$) is calculated. The computed gradient will be the maximum dissimilarity among all pairwise dissimilarities:')

st.latex(r'''\nabla_B^T(f)(v) = \bigvee_{i,j \in B_v} d_n(T_i, T_j)''')

st.write('There are several dissimilarity measures proposed for DTI applications and explored through the TMG (...)')

st.write('The choice of the structuring element generally depends on the information required (bi- or three-dimensional) and the size of the studied structure (...)')

#---------------------------------------

st.header('Dissimilarity measures')

#-----------------

st.subheader('Dot Product (prod)')

st.write('Only takes in consideration the main eigenvectors. The result varies between 0 and 1 according to the angle between the vectors: equals to one if the vectors are\
          parallel and equals to zero if the vectors are orthogonal.')

st.latex(r'''d_{DP}(T_i, T_j) = |e_{1i} \cdot e_{1j}|''')

st.write('Since the Dot Product is a similarity measure, the TMG calculates the minimal similarity instead of the maximal dissimilarity. The negative of the resulting image\
          will be the output of the TMG (maintaining the same pattern as for the other measures).')

#-----------------

st.subheader('Frobenius Norm (frob)')



#-----------------

st.subheader('J-Divergence (Jdiv)')



#-----------------

st.subheader('Log-Euclidean distance (logE)')

st.write('The LogE is basically an Euclidean distance measure between the logarithm of the tensors, as follows:')

st.latex(r'''d_{LE}(T_i, T_j) = \sqrt{trace((\log(T_i)-\log(T_j))^2)}''')

st.write('From the equation above, it is possible to see that symmetric matrices with null or negative eigenvalues are at an infinite distance from any tensor.\
          To overcome this problem, $\log(T_i)$ was replaced with $E_i \cdot \log(100\lambda_i+1)I\cdot E_i^T$, where $E$ corresponds to the matrix of eigenvectors,\
          $\lambda$ corresponds to the eigenvalues, and $I$ is the identity matrix.')

#---------------------------------------

st.header('Structuring elements')

#...

st.subheader('Visualization')