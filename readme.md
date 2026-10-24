# Bearing Fault Diagnosis Model Based on Multiple Synchronous Compression Transformation MSST+CNN

\tPreface
This update introduces a new product based on multiple synchronous compression transformations.
\t(Multisynchrosqueezing Transform, MSST) and CNN-based bearing fault diagnosis method, combining advanced time-frequency analysis technology with deep learning effectively solves the problem of fault feature extraction under variable working conditions.
Traditional FFT methods, based on the assumption of signal stationarity, struggle to capture instantaneous frequency changes in bearing vibration signals, leading to vague fault characteristics. Wavelet Transform (CWT) suffers from energy diffusion issues and limited time-frequency resolution. MSST enhances energy concentration and time-frequency resolution through synchronous compression techniques, enabling clear visualization of the time-frequency trajectory of impact components. By converting one-dimensional vibration signals into high-resolution MSST time-frequency images, it fully leverages the powerful feature extraction capabilities of Convolutional Neural Networks (CNN), ensuring both local time-frequency features while significantly improving the model's adaptability to non-stationary signals. (Not only does this provide step-by-step tutorials for project use, but also includes video explanations.) For a detailed introduction to the University of Kansas' bearing data, refer to the following article:
Python-Case Western Reserve University (CWRU) Bearing Data Interpretation and Classification Process
● Dataset: CWRU Bearing Fault Dataset
● Environment Framework: Python 3.11, PyTorch 2.1 and above can be run.
● Accuracy: 99.79% on the test set
● Usage: Beginners, those in need of papers, graduate students
● Code Guarantee: The code comments are detailed and ready to run.
1. Project Overview
\tCore idea
(1) Time-frequency Transformation: Utilizes Multi-Resolution Synchronous Compression Transform (MSST) to convert one-dimensional vibration signals into high-resolution time-frequency images. By rearranging the time-frequency coefficients, it enhances energy concentration, clearly displays the time-frequency trajectory of shock components, overcoming the limitations of traditional FFT in processing non-stationary signals and the issue of energy diffusion in wavelet transforms.
(2) Feature Extraction: By utilizing Convolutional Neural Networks (CNN), deep discriminative features of time-frequency images are automatically extracted. This process fully exploits the time-frequency local information related to faults, transforming manually designed features into hierarchical representations learned by the network.
(3) Fault Identification: By mapping the extracted image features through a fully connected layer to the bearing fault categories, it realizes an end-to-end intelligent diagnosis from raw vibration signals to diagnostic results.
2. Bearing Data Loading and Preprocessing
2.1 Read data and organize into csv file
2.2 Algorithm
2.2.1 Data Preprocessing
In the first step of data preprocessing, we need to convert the raw MATLAB format bearing vibration data into a conveniently processed CSV format. This step is implemented by running dataset_make_csv.py file, which uses scipy.io.loadmat to read the vibration signals from the drive end of the CWRU bearing dataset in 12k files. The original data are distributed across multiple .mat files according to different fault types and rotational speeds, with each speed corresponding to 10 .mat files, representing normal conditions and three types of fault positions (inner ring, rolling element, outer ring) under three damage sizes (7, 14, 21 mil), totaling 10 categories. The script processes each speed separately, with each speed corresponding to 10 .mat files, extracting the vibration data column for the drive end (such as X097_DE_time) from each file, unfolding the data into a one-dimensional array, and uniformly clipping the first 119808 sampling points to ensure consistent sample lengths. Subsequently, these data are stored in a Pandas DataFrame according to predefined column names (such as normal, diameter_7_inner, etc.), using the first column 'normal' as an index, and finally saving the DataFrame as a CSV file. After completion, four CSV files will be generated: data_12k_1797.csv, data_12k_1772.csv, data_12k_1750.csv, and data_12k_1730.csv, each containing 10 columns of vibration data for subsequent time-frequency analysis and model training.
The MSST (Multiple Synchronous Compression Transform) overcomes the limitations of FFT in non-stationary signal processing by addressing the issues of energy diffusion and low resolution in wavelet transform. It offers higher energy concentration and time-frequency resolution, enabling clear visualization of instantaneous frequency changes and shock characteristics in signals.

The algorithm part of MATLAB code is as follows:
This function implements the Multiple Synchronous Compression Transform (MSST), receiving three input parameters: a one-dimensional column vector signal x, the length of the window function hlength, and the number of iterations num, returning three outputs: the final Multiple Synchronous Compression Time-Frequency Matrix Ts, the Short-Time Fourier Transform result tfr, and the instantaneous frequency matrix omega2 after multiple iteration refinement.
Preprocessing the window function: The window length is forced to be odd to ensure symmetry, generating a symmetric time vector ht, and constructing a Gaussian window h (window function form exp(-π/0.32²·t²)) based on this. The half length Lh of the window is calculated for subsequent signal boundary handling. At the same time, obtain the signal length N and time index t, and initialize matrices for storing STFT results, instantaneous frequencies, post-iteration frequencies, and the final MSST results. All matrices have rows that are rounded to round(N/2), i.e., only retain positive frequencies.
Perform energy redistribution: Based on the refined frequency matrix omega2, reallocate the energy of each time-frequency point to its corresponding target frequency slot. Specifically, for each time-frequency point (η, b), if its STFT magnitude is greater than a set minimal threshold to prevent noise interference, then retrieve its target frequency index k = omega2(η, b). If k is within the valid range, then accumulate the complex value tfr(η, b) at the current point to Ts(k, b). After reassigning all points, obtain the final multiple synchronous compression time-frequency representation Ts. Finally, normalize both tfr and Ts (divided by xrow/2) to ensure an appropriate energy scale, and the function ends.

2.3 Generate time-frequency plot
Convert a one-dimensional vibration signal of length 1024 into a multiple synchronous compression transformation (MSST) time-frequency image suitable for convolutional neural network (CNN) training. First, call the MSST_Y_new function to calculate the MSST time-frequency matrix MTs of the signal, which reflects the energy distribution in the time-frequency domain; then take the absolute values of the matrix to obtain magnitudes and normalize them to the [0,1] interval for subsequent image processing; next, convert the normalized data into grayscale indexes ranging from 0 to 255 through im2uint8, and use jet color map to map it into pseudo-color RGB images, making the time-frequency features more easily recognizable in visual terms; finally, resize the image to 224×224 pixels (the standard size for CNN input), save it as a PNG file, completing the conversion from raw vibration data to model input image.
1. Batch generation of MSST time-frequency plots. This part of the code traverses the list of signal segments rearranged in a random order, converting each vibration signal segment into a time-frequency image and automatically assigns them to corresponding folders based on the preset training, validation, and test set proportions. The loop variable i ranges from 1 to num_signals. First, it retrieves the current signal segment signal_segment from the cell array signal_segments; then, based on the relationship between i and the number of training sets num_train, validation sets num_val, determines its classification: if i <= num_train, it is classified as a training set and generates a filename with the format condition_name_sequence_number.png, where condition_name represents the current fault type (such as diameter_7_inner), train_count is the count of the current fault type that has been processed in the training set; similarly, when i is between num_train+1 and num_train+num_val, it is classified as a validation set, otherwise it is classified as a test set. After determining the filename, it calls the custom function image_generate to perform multiple synchronous compression transformations (MSST) on the signal segment and saves it as a corresponding time-frequency image, while also outputting the current generated file path through fprintf for real-time monitoring of processing progress.
\t  
The partial time-frequency diagram is shown in the figure below, which indicates that compared to the wavelet transform MSST, it possesses higher resolution.
\t  
3. Fault Diagnosis Model Based on MSST+CNN
3.1 Define the network model, set parameters, and train the model
In the construction of deep learning models, data loading is a crucial step. In this project, we have customized the `CWRUDataset` class, which inherits from PyTorch's `Dataset` base class, to efficiently read and manage bearing fault frequency image data. During its initialization process, it is necessary to specify the root directory `root_dir` for the images and an optional preprocessing operation `transform`. At initialization, it will traverse all files in the root directory and filter out common image formats (such as PNG, JPG, JPEG). Based on the preset file naming rule "speed_fault_type_serial_number.png", each valid file is parsed according to the extracted fault type field from the filename. After extracting the fault type field, it matches with the predefined label mapping dictionary `label_map` to convert the specific fault type (such as `diameter7inner`) into corresponding numerical labels (0-9). Finally, the complete path of each image and its corresponding label are combined into a tuple, stored in the `self.samples` list, to prepare for subsequent batch reading and model training.
After completing data preparation, the next step is to construct a convolutional neural network (CNN) model for fault diagnosis. In this project, a network structure named `SimpleCNN` is defined, which inherits from the PyTorch `nn.Module` base class and adopts a classic design paradigm of convolutional neural networks. It consists of two core parts: feature extraction and classification.

In the feature extraction phase, the network progressively learns time-frequency features related to faults through four convolutional blocks. Each block consists of a convolution layer, ReLU activation function, and max pooling layer. The input image is an RGB three-channel 224×224 time-frequency map. The first convolutional block uses 32 3×3 convolutional kernels to extract shallow features, with dimensions maintained by padding, followed by a 2×2 max pooling that halves the feature map size. The subsequent three convolutional blocks sequentially expand the number of channels to 64, 128, and 256, with each block reducing the feature map size by half again, eventually resulting in a 14×14×256 feature map. This hierarchical deepening design allows the abstraction of high-level fault semantic information from low-level edge and texture features.

The extracted feature maps from the convolutional layers are then fed into an adaptive average pooling layer, which compresses each channel's feature map to 1×1, resulting in a 256-dimensional feature vector. This vector is then unfolded and input into the classification part. The classifier consists of two fully connected layers, with ReLU activation functions and Dropout layers inserted between them to prevent overfitting. The first fully connected layer maps the 256-dimensional feature to 512-dimensional, after which nonlinear activation occurs. Dropout randomly drops neurons with a probability of 0.5 to enhance the model's generalization ability. The last fully connected layer maps the 512-dimensional feature to 10-dimensional outputs, corresponding to normal conditions and nine types of fault categories.
In the forward propagation process, the input image is successively processed through feature extraction, adaptive pooling, flattening operations, and a classifier, ultimately outputting the predicted scores for each category, which are used for subsequent loss calculation and classification. The entire model structure is simple and efficient, fully reflecting the typical design philosophy of convolutional neural networks in image classification tasks.
Model Evaluation:
1. Only ten rounds of training were conducted, with the final test results: Loss: 0.0506, Accuracy: 0.9833
2. Fault-specific confusion matrix:
3. Training data t-SNE feature visualization:
Click to download: Complete original data, Python code
Previously interesting content:
Python Bearing Fault Diagnosis | From Beginner to Publication
Python Bearing Fault Diagnosis (14) High Innovation Fault Identification Model
SCI 1 Category High Innovation Bearing Fault Diagnosis Model!
Attention Magic Modification | Super Bearing Fault Diagnosis Model!
Bearing Fault Diagnosis Model Based on K-NN + GCN
27 Bearing Fault Diagnosis Models | The most cost-effective family set available online
| Based on KAN, KAN Convolutional Bearing Fault Diagnosis Model
Time-frequency graph update | Bearing Fault Diagnosis based on Visual Top Conference + Time-frequency graph
Fault Diagnosis | Code explanation + live courses, hotly ongoing!
Rapid publication papers | Based on 2D-SWinTransformer+1D-CNN-SENet parallel fault diagnosis model
Positional encoding demystification | Explanation of Transformer's positional encoding Positional Encoding
Resource sharing | Intelligent fault diagnosis and prediction based on graph neural networks
Innovation points | Fast Fourier Transform Convolution (FFC) based fault diagnosis model
Code open-source! | Field adversarial graph convolution network fault diagnosis under variable conditions
Over 30 open-source datasets | No more worries about data sets for fault diagnosis!
Fault Diagnosis | A must-have for researchers in their first and second year of graduate studies!!!
Superior! Top conference innovation integration! Parallel classification network based on 2D-SWinTransformer
Multimodality - Fault Diagnosis | Large convolutional kernels open a new era in visuals!
The strength of the Transformer structure, How much attention do you need?
Fault Diagnosis | A small innovation: feature extraction + KAN classification
Superior! Directly write in the first area! Based on SSA+Informer-SENet fault diagnosis model
Fault Diagnosis | Collection of signal noise reduction algorithms
Graph convolutional fault diagnosis, newly added GAT, SGCN, GIN classification models
Bearing life prediction kit is here!
Cannot miss out! A major update in fault diagnosis + time-frequency image classification!
Recommended journals for intelligent fault diagnosis and life prediction.
Fault Diagnosis directly written in the first area, graph convolution + BiGRU-Attention parallel diagnostic model
END

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
