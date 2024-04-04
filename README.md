## TropAI: Advancing Torah Cantillation with AI

### Introduction
TropAI is a pioneering project designed to harness the power of artificial intelligence for the enhancement of Torah cantillation learning. Through sophisticated analysis of audio recordings, it aims to identify and teach Taamei Hamikra (cantillation marks) with unparalleled precision. This presentation delineates the methodologies TropAI employs, focusing on pitch extraction with the YIN algorithm and sequence learning through Recurrent Neural Networks (RNN) equipped with GRU cells.

### The Challenge
The intricate practice of Torah cantillation, marked by unique cantillations for syntactic and musical guidance, poses a significant learning curve. Achieving proficiency requires nuanced feedback that traditional learning methods may not adequately provide.

### TropAI's Technological Approach
TropAI's innovative solution leverages cutting-edge technology to address the nuances of cantillation:

1. **YIN Algorithm for Pitch Extraction**:
   - **Benefit**: The YIN algorithm is chosen for its exceptional balance of computational efficiency and accuracy in detecting pitch, crucial for distinguishing the subtle variations in cantillation.
   - **Application**: Facilitates the initial step of cantillation analysis by accurately identifying pitch from audio inputs with minimal computational resources.

```python
# Simplified implementation of the YIN algorithm
import numpy as np

def difference_function(x, N):
    delta = np.zeros(N // 2)
    for tau in range(1, N // 2):
        delta[tau] = np.sum((x[:N] - x[tau:N + tau])**2)
    return delta

def cumulative_mean_normalized_difference_function(delta, N):
    cmndf = np.zeros(N // 2)
    cmndf[1] = 1
    for tau in range(2, N // 2):
        cmndf[tau] = delta[tau] * tau / np.sum(delta[1:tau + 1])
    return cmndf

def absolute_threshold(cmndf, threshold=0.1):
    for tau in range(2, len(cmndf)):
        if cmndf[tau] < threshold and cmndf[tau] < cmndf[tau + 1]:
            return tau
    return -1

def yin_pitch_detection(audio_signal, sr, w_size=2048, hop_size=1024, threshold=0.1):
    pitches = []
    frames = range(0, len(audio_signal) - w_size, hop_size)
    for frame in frames:
        x = audio_signal[frame:frame + w_size]
        N = len(x)
        delta = difference_function(x, N)
        cmndf = cumulative_mean_normalized_difference_function(delta, N)
        tau = absolute_threshold(cmndf, threshold)
        if tau != -1:
            f0 = sr / tau
            pitches.append(f0)
        else:
            pitches.append(0)
    return pitches
```

2. **RNN with GRU for Taamim Recognition**:
   - **Benefit**: GRU cells are selected for their efficiency and capability in learning from sequences, making them ideal for decoding the complex patterns of taamim from pitch data. GRUs provide a streamlined architecture that maintains performance while reducing training time and computational demands.
   - **Application**: Empowers the model to learn and predict sequences of taamim, offering feedback based on the audio's cantillation patterns.

```python
# RNN model using GRU cells for taamim recognition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed

def build_model(input_shape, num_classes):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        GRU(128, return_sequences=True),
        TimeDistributed(Dense(num_classes, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

3. **Data Augmentation for Robust Learning**:
   - **Benefit**: Data augmentation techniques, such as pitch shifting and speed variation, artificially expand the dataset, allowing the model to learn from a broader array of cantillation styles and conditions. This ensures robustness against variations in voice, tempo, and recording quality.
   - **Application**: Enhances the training dataset, preparing TropAI to accurately recognize taamim across diverse real-world scenarios.

```python
# Augmenting audio data for diverse training inputs
import librosa

def augment_audio(y, sr, pitch_shift_steps=0, speed_factor=1.0):
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=pitch_shift_steps)
    y_stretched = librosa.effects.time_stretch(y_shifted, speed_factor)
    return y_stretched
```

### Data Strategy

TropAI's strategic data creation and augmentation ensure a dataset rich in diversity. By focusing on the most common sequences of taamim and then generating sentences with varying Hebrew words to match these cantillations, we're ensuring that the model learns the cantillation patterns themselves, rather than memorizing specific word-cantillation combinations. This method also helps in generalizing across different texts and speakers, a crucial factor for robust model performance. By recording sentences with varied Hebrew words and applying pitch and speed augmentations, TropAI prepares to handle real-world variations in cantillation.

### Future Directions

TropAI, currently in development and testing, plans to enhance dataset diversity, improve model accuracy, and refine user experience. The goal is to provide an accessible tool for mastering Torah cantillation, benefiting learners at all levels.

### Conclusion
TropAI embodies the synergy of traditional Torah study and modern artificial intelligence, offering a novel

 approach to mastering cantillation. By leveraging the YIN algorithm and GRU-based RNNs, alongside strategic data augmentation, TropAI is poised to become an invaluable tool for learners seeking to deepen their practice and understanding of Torah cantillation. Through continuous development and adaptation, TropAI aspires to set new standards in the digital enhancement of religious study.
