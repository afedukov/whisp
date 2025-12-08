# 🎙️ Whisper Transcription Tool

Beautiful command-line application for transcribing audio files using [OpenAI Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3) model.

Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for **4-8x faster** transcription compared to standard implementations.

## ✨ Features

- 🎯 High accuracy transcription with Whisper Large-v3
- ⚡ **4-8x faster** than standard Whisper (using CTranslate2)
- 🚀 GPU (CUDA) support for accelerated processing
- 💾 Lower memory usage with int8 quantization on CPU
- 🌍 Automatic language detection or manual language specification
- 📊 Beautiful progress indicators in Claude Code style
- 📝 Preview of transcription results
- 🔄 Multiple model options (large, turbo, medium, small)
- 🎤 Voice activity detection (VAD) to skip silence

## 🎵 Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- Other formats supported by ffmpeg

## 📋 Requirements

- **Python 3.11, 3.12, or 3.13** (⚠️ Python 3.14 not supported yet due to dependencies)
- ffmpeg (for audio processing and format conversion)

### Installing Python 3.11

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

**Windows:**
Download Python 3.11 from [python.org](https://www.python.org/downloads/) and install.

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

### 2. Create a virtual environment with Python 3.11

**macOS/Linux:**
```bash
python3.11 -m venv venv
```

**Windows:**
```bash
python -m venv venv
```

> 💡 **Tip:** Make sure you're using Python 3.11-3.13. Check with `python3.11 --version`

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

This will install:
- `faster-whisper` - Optimized Whisper implementation (4-8x faster)
- `ctranslate2` - Inference engine for transformer models
- `rich` - Beautiful terminal formatting
- Other required dependencies

> ⚠️ **Note:** The first run will take some time as the Whisper model (~3GB for large) will be downloaded. The model is cached locally for future use.

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
│ Whisper Transcription Tool              │
│ Powered by OpenAI whisper-large-v3      │
╰─────────────────────────────────────────╯

Device: cpu

Initializing Whisper model...
Model: turbo (Best accuracy, ~3GB (uses large-v3 with optimizations))
Using faster-whisper for optimized performance
✓ Model loaded successfully on cpu
Compute type: int8

Converting .m4a to WAV format...
✓ Audio converted successfully

Transcribing audio file...
Input: lecture.m4a
Duration: 08:03

⠹ Processing 08:03 of audio... 0:02:15
Detected language: de (confidence: 99.8%)

Saving transcription...
✓ Transcription saved to: transcript.txt

Preview:
╭─────────────────────────────────────────╮
│ Willkommen zu dieser Vorlesung über    │
│ kognitive Psychologie. Heute werden... │
╰─────────────────────────────────────────╯

Stats: 1247 words, 7856 characters

Transcription completed successfully!
```

## ⚡ Performance

Thanks to `faster-whisper` with CTranslate2 and int8 quantization:

| Setup | Speed | Example (8 min audio) |
|-------|-------|----------------------|
| **CPU (int8)** | ~0.5-1x real-time | **2-4 minutes** ⚡ |
| **GPU (CUDA)** | ~10-20x real-time | **30-60 seconds** 🚀 |

**Comparison:**
- Standard Whisper (transformers): 16-40 minutes for 8min audio on CPU
- **faster-whisper (this tool): 2-4 minutes** for same audio ✅
- **4-8x faster** than standard implementation!

> 💡 **Tip:** Even on CPU, faster-whisper provides excellent performance thanks to int8 quantization and optimized inference.

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

### Python version compatibility error

If you see errors about `onnxruntime` or dependency conflicts:

```bash
# Check your Python version
python --version

# Should be 3.11.x, 3.12.x, or 3.13.x
# If you have Python 3.14, you need to use Python 3.11-3.13
```

**Solution:** Recreate your venv with Python 3.11:
```bash
# Remove old venv
rm -rf venv

# Create new venv with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### "ffmpeg not found" error

Make sure ffmpeg is installed and available in PATH:

```bash
ffmpeg -version
```

If not installed:
- **macOS:** `brew install ffmpeg`
- **Ubuntu:** `sudo apt install ffmpeg`
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### "Cannot install faster-whisper" error

This usually happens with Python 3.14+. Make sure you're using Python 3.11-3.13:

```bash
python --version  # Should show 3.11.x, 3.12.x, or 3.13.x
```

### Slow performance

- faster-whisper with int8 quantization on CPU should process 8min audio in ~2-4 minutes
- If it's much slower, check that `faster-whisper` is actually installed:

```bash
python -c "from faster_whisper import WhisperModel; print('OK')"
```

### GPU not detected

To use GPU acceleration, you may need to install CUDA-enabled dependencies. Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
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

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper) - The optimized implementation we use
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference engine for Transformer models
- [Whisper Large-v3 on HuggingFace](https://huggingface.co/openai/whisper-large-v3)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper) - Original Whisper repository
- [Whisper Model Card](https://github.com/openai/whisper/blob/main/model-card.md) - Technical details and benchmarks

---

**Made with ❤️ using OpenAI Whisper and faster-whisper**
