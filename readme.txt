Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,? Information Fusion, vol. 39, pp. 168?177, 2018.
mssc_test(Ytot,Label)

and

mlrr_test(Ytot,Label)

are demo test.  A test dataset (AR dataset and its facial components) is availabl for test.
Simply load:

load('ARObjects.mat')

and test as:

mssc_test(Ytot,Label)
mlrr_test(Ytot,Label)


inputs for the demo functions:


  Inputs:
      Ytot:  A cell type variable containing modalities matrices in each
      cell.
              Ytot{1}= Modality_1 \in R^{D1xN}
              Ytot{2}= Modality_2 \in R^{D2xN}
              ...

      Label:  An N-sized vector containing the labels for the data points
              (is only used for evaluation).

 Output:   The function displays the clustering performances in terms of
           error rate and NMI.