# Compound Fault Diagnosis for Train Transmission Systems Using Deep Learning with Fourier-enhanced Representation
**Paper**: https://arxiv.org/abs/2504.07155

Fault diagnosis prevents train disruptions by ensuring the stability and reliability of their transmission systems. Data-driven fault diagnosis models have several advantages over traditional methods in terms of dealing with non-linearity, adaptability, scalability, and automation. However, existing data-driven models are trained on separate transmission components and only consider single faults due to the limitations of existing datasets. These models will perform worse in scenarios where components operate with each other at the same time, affecting each component's vibration signals. To address some of these challenges, we propose a frequency domain representation and a 1-dimensional convolutional neural network for compound fault diagnosis and applied it on the PHM Beijing 2024 dataset, which includes 21 sensor channels, 17 single faults, and 42 compound faults from 4 interacting components, that is, motor, gearbox, left axle box, and right axle box. Our proposed model achieved 97.67% and 93.93% accuracies on the test set with 17 single faults and on the test set with 42 compound faults, respectively.

![FFT-1DCNN framework](https://github.com/user-attachments/assets/07383c10-c6af-4277-b23b-cf7433aecad1)

**Paper Citation**: Rico, J. A., Raghavan, N., & Jayavelu, S. (2025). Compound Fault Diagnosis for Train Transmission Systems Using Deep Learning with Fourier-enhanced Representation. arXiv preprint arXiv:2504.07155.

**Dataset**: https://2024.icphm.org/datachallenge/

**Dataset Citation**: Ding, A., Qin, Y., Wang, B., Guo, L., Jia, L., & Cheng, X. (2024). Evolvable graph neural network for system-level incremental fault diagnosis of train transmission systems. Mechanical Systems and Signal Processing, 210, 111175.


