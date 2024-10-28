Siamese Triplet Network for Audio Classification

This project explores the challenges of applying deep metric learning to Out-of-Distribution (OOD) data using a Siamese neural network implementation that performs few-shot learning on the Spoken MNIST dataset, which consists of audio recordings of spoken digits converted into spectrograms. The architecture serves as a feature extractor, trained on digits 0-7 using triplet loss to learn an embedding space where similar samples cluster together while dissimilar ones are pushed apart. The model demonstrates strong generalization by effectively clustering unseen digits (8 and 9) in the embedding space, showing how learned audio-visual patterns can be leveraged for few-shot classification of novel classes.

An Ilustration of Siamese Triplet Loss Architecture:
<img width="601" alt="image" src="https://github.com/user-attachments/assets/41712434-3efa-4f4e-9829-f4858f405c5a">




Credits: deeplake, activeloop
