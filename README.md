# Signal-Denoising-Using-Transformer-model
🔍 Overview
The Signal Denoising Transformer (DeT) is a deep learning model designed to remove noise from one-dimensional biosignals (e.g., EEG, EMG). By breaking the input sequence into fixed-length patches, embedding them, and applying multi-head self-attention across time, DeT learns long-range dependencies and accurately reconstructs the underlying clean waveform.

🚀 Tools & Technologies
Frameworks: PyTorch (modeling & training), NumPy (data handling), Matplotlib (visualization)

Libraries: einops (patch embedding), tensorboardX (training logs), tqdm (progress bars)

Hardware: NVIDIA GPUs with CUDA (accelerated training)

📁 Files & Structure
model.py

Transformer block (stacked nn.TransformerEncoderLayer)

DeT class (patch embedding → positional dropout → Transformer → patch reconstruction)

data_prepare.py

Functions to load, slice, and normalize clean/noisy EEG or EMG .npy files into train/test splits

train.py

Training loop (1,000 epochs) with MSE loss, Adam optimizer, learning-rate scheduler

Checkpointing (best.pth, periodic epoch saves) and TensorBoard logging

evaluate.py

Metric functions: temporal RRMS, spectral RRMS, correlation coefficient

Inference on held-out test set, loss tracking, waveform/spectrogram plots

📈 Training & Testing
Dataset:

Clean EEG signals (EEG_all_epochs.npy) paired with synthetic or recorded noisy versions (EMG_all_epochs.npy)

Split: 80% training, 20% validation

Hyperparameters:

Epochs: 1,000

Batch size: 100

Sequence length: 512 samples

Patch length: 64 (→ 8 patches per sequence)

Transformer depth: 6 layers

Heads: 4

Dropout: 0.1

Results:

Training Loss (MSE): converged to ~0.001

Validation Loss: stabilized around ~0.002 after epoch 800

Temporal RRMS: reduced from ~1.2 (initial) to ~0.35

Spectral RRMS: < 0.25

Correlation Coefficient: increased from ~0.02 to ~0.88

🌐 Real-World Applications

Clinical EEG/EMG Denoising: Improves signal quality for brain–computer interfaces, epilepsy monitoring, sleep studies

Wearable Sensors: On-device noise suppression for portable health monitors (e.g., muscle fatigue or stress detection)

Industrial IoT: Cleaning vibration or acoustic sensor data in manufacturing for predictive maintenance

Audio Restoration: Adapting architecture for speech/music denoising in telecommunication or media production
