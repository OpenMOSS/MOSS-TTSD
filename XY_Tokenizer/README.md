# XY Tokenizer

XY Tokenizer is a speech codec that simultaneously models both semantic and acoustic aspects of speech, converting audio into discrete tokens and decoding them back to high-quality audio. It achieves efficient speech representation at only 1kbps with RVQ8 quantization at 12.5Hz frame rate.

## Features

- **Dual-channel modeling**: Simultaneously captures semantic meaning and acoustic details
- **Efficient representation**: 1kbps bitrate with RVQ8 quantization at 12.5Hz
- **High-quality audio tokenization**: Convert speech to discrete tokens and back with minimal quality loss
- **Long audio support**: Process audio files longer than 30 seconds using chunking with overlap
- **Batch processing**: Efficiently process multiple audio files in batches
- **32kHz output**: Generate high-quality 32kHz audio output

## Installation

```bash
# Create and activate conda environment
conda create -n xy_tokenizer python=3.10 -y && conda activate xy_tokenizer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Inference

To tokenize audio files and reconstruct them:

```bash
python inference.py \
  --config_path ./config/xy_tokenizer_32k_config.yaml \
  --checkpoint_path ./weights/xy_tokenizer.ckpt \
  --input_dir ./input_wavs/ \
  --output_dir ./output_wavs/
```

### Parameters

- `--config_path`: Path to the model configuration file
- `--checkpoint_path`: Path to the pre-trained model checkpoint
- `--input_dir`: Directory containing input WAV files
- `--output_dir`: Directory to save reconstructed audio files
- `--device`: Device to run inference on (default: "cuda")
- `--debug`, `--debug_ip`, `--debug_port`: Debugging options (disabled by default)

## Project Structure

- `xy_tokenizer/`: Core model implementation
  - `model.py`: Main XY_Tokenizer model class
  - `nn/`: Neural network components
- `config/`: Configuration files
- `utils/`: Utility functions
- `weights/`: Pre-trained model weights
- `input_wavs/`: Directory for input audio files
- `output_wavs/`: Directory for output audio files

## Model Architecture

XY Tokenizer uses a dual-channel architecture that simultaneously models:
1. **Semantic Channel**: Captures high-level semantic information and linguistic content
2. **Acoustic Channel**: Preserves detailed acoustic features including speaker characteristics and prosody

The model processes audio through several stages:
1. Feature extraction (mel-spectrogram)
2. Parallel semantic and acoustic encoding
3. Residual Vector Quantization (RVQ8) at 12.5Hz frame rate (1kbps)
4. Decoding and waveform generation

## License

[Specify your license here]