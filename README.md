# 🎙️ Whisper Transcription Tool

Beautiful command-line application for transcribing audio files using [OpenAI Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3) model.

## ✨ Features

- 🎯 High accuracy transcription with Whisper Large-v3
- 🚀 GPU (CUDA) support for accelerated processing
- 🌍 Automatic language detection or manual language specification
- 📊 Beautiful progress indicators in Claude Code style
- 📝 Preview of transcription results
- 💾 Save transcription to text file
- 🔄 Multiple model options (large, turbo, medium, small)

## 🎵 Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- Other formats supported by ffmpeg

## 📋 Requirements

- Python 3.8 or higher
- ffmpeg (for audio processing)

### Installing ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## 🚀 Installation

### 1. Clone the repository or copy the files

```bash
cd whisp
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note:** The first run will take some time as the Whisper model (~3GB for large) will be downloaded from HuggingFace.

## 💻 Usage

### Basic usage

```bash
python whisp.py input.mp3 output.txt
```

### With model and language specification

```bash
python whisp.py audio.wav transcript.txt --model large --language en
```

### Model Selection

4 Whisper models are available:

| Model | Size | Accuracy | Speed | Recommendation |
|-------|------|----------|-------|----------------|
| **large** | ~3GB | Best | Slow | ✅ Default, for academic content |
| **turbo** | ~1.5GB | Same as large | 8x faster | ⚡ Recommended for most tasks |
| **medium** | ~1.5GB | Good | 2-3x faster | ⚖️ Balance of speed and quality |
| **small** | ~466MB | Basic | Fast | 🚀 For simple tasks |

### Examples

**German lecture (maximum accuracy):**
```bash
python whisp.py lecture.mp3 transcript.txt --model large --language de
```

**Fast podcast transcription:**
```bash
python whisp.py podcast.m4a transcript.txt --model turbo --language en
```

**Automatic language detection:**
```bash
python whisp.py interview.mp3 interview_text.txt --model medium
```

### Command help

```bash
python whisp.py --help
```

## 📊 Example Output

```
╭─────────────────────────────────────────╮
│                                         │
│  🎙️  Whisper Transcription Tool        │
│  Powered by OpenAI Whisper Large-v3    │
│                                         │
╰─────────────────────────────────────────╯

Device: cuda:0
GPU: NVIDIA GeForce RTX 3080

⚙️  Initializing Whisper model...
Model: turbo (Same accuracy as large, 8x faster, ~1.5GB)
✓ Model loaded successfully on cuda:0

🔧 Creating transcription pipeline...
✓ Pipeline ready

🎵 Transcribing audio file...
Input: audio.mp3

⠋ Processing audio... ━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:15

💾 Saving transcription...
✓ Transcription saved to: transcript.txt

📝 Preview:
╭─────────────────────────────────────────╮
│                                         │
│  Hello and welcome to today's podcast. │
│  In this episode, we'll be discussing...│
│                                         │
╰─────────────────────────────────────────╯

Stats: 523 words, 3142 characters

✨ Transcription completed successfully!
```

## ⚡ Performance

- **With GPU (CUDA):** ~10-20x faster than real-time
- **With CPU:** ~2-5x slower than real-time

> 💡 **Tip:** For long audio files, using GPU is highly recommended.

## 🔧 Additional Settings

### Model Selection Recommendations

- **large** - use for academic lectures, medical recordings, technical documentation
- **turbo** - optimal choice for most tasks: podcasts, interviews, meetings
- **medium** - for fast processing of simple content on weaker machines
- **small** - only for simple recordings with good audio quality

### Supported Languages

Whisper supports 99+ languages. Most popular:
- `en` - English
- `ru` - Russian
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

Full list: [Whisper Language Support](https://github.com/openai/whisper#available-models-and-languages)

## 🐛 Troubleshooting

### "CUDA out of memory" error

If you don't have enough GPU memory, reduce `batch_size` in the code:

```python
batch_size=8,  # instead of 16
```

### "ffmpeg not found" error

Make sure ffmpeg is installed and available in PATH:

```bash
ffmpeg -version
```

### Slow performance

- Make sure GPU is being used (output should show `cuda:0`)
- Check that PyTorch with CUDA support is installed:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If output is `False`, reinstall PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📦 Project Structure

```
whisp/
├── whisp.py                # Main script
├── requirements.txt         # Python dependencies
├── README.md               # Documentation
└── venv/                   # Virtual environment (created during installation)
```

## 📄 License

This project uses the Whisper model from OpenAI. See [Whisper License](https://github.com/openai/whisper/blob/main/LICENSE) for details.

## 🤝 Contributing

Questions and suggestions are welcome! Create issues or pull requests.

## 📚 Useful Links

- [Whisper Large-v3 on HuggingFace](https://huggingface.co/openai/whisper-large-v3)
- [Whisper Large-v3-Turbo on HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)

---

**Made with ❤️ using OpenAI Whisper and HuggingFace Transformers**
