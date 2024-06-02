# mumutc_MLanalysis
This program does a Machine Learning analysis by data that were used in publication [https://arxiv.org/pdf/2302.01143](https://arxiv.org/pdf/2302.01143). 
Two methods are implemented: 
* Boost Decision Tree (BDT)
* Deep Neural Network (DNN)

Models are built by using the TMVA library provided by ROOT ([https://root.cern/manual/tmva/](https://root.cern/manual/tmva/)). 
Principlely, the Deep Learning provided by Keras can be implemented, too. 
But the tensorflow interface of ROOT (version 6.30.03) does not match to the new version of tensorflow (>=2.16). 
It may work for some old tensorflow version.

**MLresults.pdf** show the results of ML analysis.
