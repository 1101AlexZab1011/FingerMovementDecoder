# Non-invasive Finger Movement Decoding with LF-CNN

Welcome to the GitHub repository for our project on non-invasive finger movement decoding using a Linear Finite Impulse Response Convolutional Neural Network (LF-CNN). In this repository, you will find the code and resources related to our research on accurately classifying finger movements from non-invasive neurophysiological recordings.

## Objective

Non-invasive Brain-to-Computer interfaces have made significant progress in accurately classifying hand movement lateralization. However, distinguishing the activation patterns of individual fingers within the same hand remains a challenging task due to their overlapping representation in the motor cortex. Our objective was to validate a compact convolutional neural network that can quickly and reliably decode finger movements from non-invasive neurophysiological recordings.

## Approach

We conducted experiments with healthy participants using Magnetoencephalography (MEG) during a serial reaction time task (SRTT), where participants pressed buttons using their left and right index and middle fingers. To achieve our goal, we employed the Linear Finite Impulse Response Convolutional Neural Network (LF-CNN) for decoding. We also compared the performance of LF-CNN with existing deep learning architectures such as EEGNet, FBCSP-ShallowNet, and VGG19.

## Results

Our research yielded the following results:

- Movement laterality was decoded with an accuracy superior to 95% by all approaches.
- For individual finger movement decoding, accuracy ranged between 80-85%.
- LF-CNN outperformed other architectures in terms of computational time.
- LF-CNN provided interpretability in both spatial and spectral domains, allowing us to examine neurophysiological patterns reflecting task-related motor cortex activity.
- The LF-CNN decoding performance was maximal when train/test trials belonged to the same phase of the task, indicating its sensitivity to cognitive learning.

## Significance

Our project demonstrated the feasibility of decoding finger movements using a tailored Convolutional Neural Network, which can dynamically track changes in neuronal activity during motor learning. The performance of LF-CNN was comparable to complex deep learning architectures while offering faster and more interpretable outcomes. This algorithmic strategy holds high potential for investigating the mechanisms underlying non-invasive neurophysiological recordings in cognitive neuroscience.

## Repository Contents

This repository contains the following resources:

- **Code**: The code for implementing LF-CNN and other deep learning architectures, as well as data preprocessing and analysis scripts.
- **Data**: Sample datasets or instructions on how to obtain the data used in our experiments.
- **Results**: Visualizations and analysis of the results obtained from the experiments.
- **Documentation**: Additional documentation, including user guides and tutorials.
- **License**: Information about the license under which this code is distributed.

## Getting Started

To get started with this project, please refer to the documentation provided in the respective folders. You may also need to install specific Python libraries and dependencies, which will be mentioned in the documentation.

## Citation

If you use this code or our research findings in your work, please cite our paper.

## Contact

If you have any questions, suggestions, or issues related to this repository, please feel free to contact us at <alexey.zabolotniy.main@yandex.ru>, <fedele.tm@gmail.com>.

Thank you for your interest in our research!
