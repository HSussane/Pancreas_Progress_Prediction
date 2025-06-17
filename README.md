# Pancreas_Progress_Prediction
Early and accurate prediction of pancreatic cancer progression is critical for optimizing treatment regimens and improving patient prognosis, particularly in patients who do not exhibit significant remission during chemotherapy. This study aimed to develop a deep learning-based longitudinal disease monitoring model using multimodal data from routine clinical practice to predict whether pancreatic cancer patients will experience disease progression at their next follow-up during chemotherapy. 
In this paper, a multimodal deep learning models is developed to predict disease progression using baseline clinical information and longitudinal CT imaging data from follow-up visits. This paper proposes a feature extractor and integrator architecture to effectively characterize the primary lesion of pancreatic cancer and its dynamic changes during follow-up using a LSTM-based approach. Specifically, the feature extractor employs a pre-trained convolutional neural network for extracting 2D CT image features, while the integrator is responsible for fusing CT image features from multiple time points and introducing clinical information into the sequence model to improve prediction accuracy. 
The study cohort included patients with advanced pancreatic ductal adenocarcinoma receiving first-line chemotherapy from two tertiary medical centers. In the first center, the training set included patients with stable disease (SD) at the first follow-up (243 patients), while the internal test set included patients with partial response (PR) at the first follow-up (41 patients). Additionally, SD and PR cases from the second center were collected as an external test set (23 patients). The optimal model was selected using five-fold cross-validation on the training set, followed by performance testing on the internal and external test sets. 
## Experiment 1 : Lesion classification
The method first constructs a lesion dataset and trains a CNN-based feature extraction network. A pre-trained CNN feature extractor is used to extract tumor imaging features from the lesion layers in the three phases of the enhanced CT scan at each time point.
The file "trainPlan2D/train_cls.py" contains the process of training the feature extractor, with the details of image acquisition and processing described in the paper.
## Experiment 2 : Progression classification
After integrating the multi-phase two-dimensional CT imaging features and performing sequential modeling, the LSTM (Long Short-Term Memory) is utilized to model these temporal imaging features in order to capture the temporal correlations between the features. Additionally, clinical information is further integrated in the feature integrator (Integrator) to enhance the predictive accuracy of the model.
The file "trainPlan2D/trian_5fold.py" contains the process of dividing the data into five folds, extracting sequential features using the extractor and integrator, and feeding these sequences into the LSTM model for training.
## Experiment 3 : Model evaluation
The file "trainPlan2D/test_outset.py" contains the similar procedures to evaluate the performance of model on an internal and external testing set.
## What is each fold for?
Folder "dataProcess": A tool package mainly contains the pre-processing of raw data with reference to the nnU-Net.
Folder "trainPlan2D": Different model training/tesing files and tool files(including sequence-processing and Extractor and Integrator modules).
## Contact information
if you have any questions, feel free to contact me. /br
Shuxiang Huang
Shenzhen University, Shenzhen, China E-mail: huangsx33333@163.com
