## CSC 466-1 Final Project Progress Log
### Quinn Coleman

#### Week 8 (Feb. 23 - 29)
Wednesday 5 - 8pm:
- Group meeting to solidify everyone on how we will implement NMF for dimensionality reduction, and the theory behind the method.

Thursday Lab Time:
- Implemented NMF in a Jupyter Notebook. Created a toy pre-processed dataset as a pandas dataframe to validate that the implementation works.

#### Week 9 (Mar. 1 - 7)
Sunday 1 - 4pm:
- Ran our NMF implementation, along with skikit-learn's NMF and PCA to reduce dimensionality of toy dataset from 6 - 2 dimensions (7 original dimensions), with 50 iterations each.
- Within these iterations, the dataset's dimensions reduced and transformed values were fetched. These transformed values were then clustered on w/ scikit-learn K-Means.
- Cluster classifications were compared with true labels of the dataset for each dimension count by accuracy, and averaged over the iterations. These accuracy scores were plotted side by side.
