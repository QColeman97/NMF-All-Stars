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

Monday 4 - 6pm:
- Encapsulated NMF function into a class, and created attributes and methods to match usage of scikit-learn's dimensionality reduction methods.
- Normalized basis vectors before updates began, and after each multiplicative update.

Tuesday 2:30 - 3pm:
- Discussed, and pivoted project analysis goal from considering performance of NMF, but instead comparing performance of NMF to another dimensionality reduction method: PCA. Relayed the pivot to the team.

Friday 1 - 2pm:
- Brainstormed about and assisted Andrew with how to make a synthetic dataset with sckikit-learn so that a weak classifier performs poorly with it.

#### Week 10 (Mar. 8 - 14)

Thursday 9pm - 1am:
- Revised experimentation with NMF: Instead of clustering the dataset at each dimensionality reduction, we now predict; this uses the dataset labels more effectively. As part of the pivot, we no longer compare our NMF to sk-learn's NMF but instead compare our NMF to sk-learn's PCA (a different method)

Friday 12:30 - :
- Made a quick performance evaluation on our NMF, to make sure it works before we start comparing it to PCA.
