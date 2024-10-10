# CNN-Siamese-Audio-Recognition
Siamese Triplet Network for Audio Classification

This project explores the challenges of applying deep metric learning to Out-of-Distribution (OOD) data using a Siamese triplet neural network. The network is designed to classify spectrograms of spoken digits, with a twist that highlights the complexities of real-world machine learning applications.

Key Features:
1. Training-Test Discrepancy:
   - Training: Spectrograms of digits 0-7
   - Testing: Spectrograms of digits 0-9, including unseen digits 8 and 9

2. Network Architecture:
   - Siamese triplet network trained using triplet loss
   - Learns to embed spectrograms into a high-dimensional space

<img width="1017" alt="image" src="https://github.com/user-attachments/assets/fd48ee63-44ff-4298-8c18-fa27b01a5ec0">





  

4. OOD Analysis:
   - In-depth examination of embedding space characteristics
   - Comparison between seen (0-7) and unseen (8-9) digits

5. Classification Strategy:
   - Utilizes Euclidean distance in the embedding space
   - Explores various distance thresholds and their impact on performance

6. Performance Trade-offs:
   - Analyzes accuracy disparities between seen and unseen digits
   - Investigates the effect of different Euclidean distance thresholds 

This project offers valuable insights into the behavior of deep learning models when faced with OOD data. It highlights the challenges of creating robust, generalizable embeddings and the complexities involved in choosing appropriate classification thresholds for both seen and unseen classes.



Credits:
deeplake
activeloop
