# Griffin Johnson 

## Project Log 1
### Week 8

#### Wed -> Group Meeting @ Library 
    - Discussed how to use NMF for dimensionality reduction. 
    - How we should measure performance. Comparing to PCA?
#### Thur -> Lab Meeting @ 466 Lab
    - Wrote pseudocode for comparing NMF to PCA by calculating pearson correlation values for basis vectors
    - Wrote pseudocode for recusrively reducing dimensions using NMF by aproximating a new V with (W)(H)=new_V
    - Discussed methodology for comparing pca and nmf for clustering
#### Sun -> Individual Work @ Home
    - Implemented previous written psuedocode for comparing NMF to PCA and recursively reducing dimensions with NMF.

### Week 9

#### Mon -> Group Meeting @ Library
    - Was able to merge key functions from respective group members code which allowed me to debug my previously written soluions with our new model for NMF
    - Discussed pre proccessing our dataset
    
## Project Log 2
### Week 9

#### Wed -> Group Meeting @ Library
    - Further discussed our changing implementation of our nmf algorithm. Adjusted recusive approach
    - Added code in our notebooks to compare mean_squared_error of sklearn's nmf and our nmf at aproximating the original input matrix
    - Added code to compare correlation at different dimension reductions between our NMF and SKlearn's nmf
    - TODO need to refactor comparing error code
    
#### Thurs -> Lab Meeting @ 466 Lab
    - Added graphing code for mean squared error and comparing pca to nmf
    - Talked about our different segments of the project and how to group together. 
    - Analysis of recursive approach to nmf
        - not working.. probably won't include
        
#### Sat -> Individual Work @ Home
    - Started code refactor
    - Adding labels and details to code.
    - Commenting for readability