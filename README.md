# 🎙️ Whisper

Command-line application for transcribing audio files using [OpenAI Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3) model (and others).

Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for **4-8x faster** transcription compared to standard implementations.

## ✨ Features

- 🎯 High accuracy transcription with Whisper Large-v3
- ⚡ **4-8x faster** than standard Whisper (using CTranslate2)
- 🚀 GPU (CUDA) support for accelerated processing
- 💾 Lower memory usage with int8 quantization on CPU
- 🌍 Automatic language detection or manual language specification
- 📊 Beautiful progress indicators with Rich
- 📝 Preview of transcription results
- 🔄 Multiple model options (large, large-v2, turbo, medium, small, base)
- 🎤 Voice activity detection (VAD) to skip silence
- 📥 Resumable model downloads with progress bar
- 🎙️ **Microphone recording mode** - record and transcribe in one command

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
- `huggingface-hub` - Model downloading with resumable transfers
- Other required dependencies

> ⚠️ **Note:** The first run will take some time as the Whisper model (~3GB for large) will be downloaded. The model is cached locally for future use. Downloads can be interrupted and resumed.

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

6 Whisper models are available:

| Model | Size | Accuracy | Speed | Recommendation |
|-------|------|----------|-------|----------------|
| **turbo** | ~800MB | Good | 8x faster | ✅ Default, fast multilingual |
| **large** | ~3GB | Best | Slow | 🎓 For academic content (latest v3) |
| **large-v2** | ~3GB | Best | Slightly faster | 🔄 Previous large version |
| **medium** | ~1.5GB | Good | 2-3x faster | ⚖️ Balance of speed and quality |
| **small** | ~466MB | Basic | Fast | 🚀 For simple tasks |
| **base** | ~145MB | Basic | Very fast | 🏃 Minimal accuracy |

### Examples

**German lecture (maximum accuracy with latest model):**
```bash
python whisp.py lecture.mp3 transcript.txt --model large --language de
```

**German lecture (maximum accuracy with large-v2):**
```bash
python whisp.py lecture.mp3 transcript.txt --model large-v2 --language de
```

**Fast podcast transcription:**
```bash
python whisp.py podcast.m4a transcript.txt --model turbo --language de
```

**Good balance for any language:**
```bash
python whisp.py interview.mp3 interview_text.txt --model medium
```

**Quick test with basic accuracy:**
```bash
python whisp.py test.mp3 test.txt --model base
```

### Recording Mode (Microphone Input)

Record audio from your microphone and transcribe it automatically.

**Basic recording:**
```bash
python whisp.py record output.txt
```

**Recording with specific model and language:**
```bash
python whisp.py record transcript.txt --model turbo --language de
```

**How it works:**
1. Shows list of available microphones
2. You select a device (or press Enter for default)
3. Press Enter to start recording
4. Press Enter again to stop recording
5. Audio is automatically transcribed using selected model
6. Transcription saved to output file
7. Temporary recording deleted (configurable in config.yaml)

**Permissions on macOS:**
- First run will ask for microphone permission
- If denied: System Settings → Privacy & Security → Microphone → Terminal

**Configuration options** (in config.yaml):
- `sample_rate`: Recording quality (default: 16000 Hz, optimal for Whisper)
- `channels`: Mono/stereo (default: 1 - mono recommended for speech)
- `keep_recording`: Keep audio file after transcription (default: false)

### Batch Mode (Directory Input)

When you provide a directory instead of a file, all audio files are processed in **natural sort order** (1, 2, 10 instead of 1, 10, 2) and combined into a single output file.

**Process all recordings in a folder:**
```bash
python whisp.py ./lectures/ combined_transcript.txt --language de --model turbo
```

**Batch mode features:**
- 📋 **Live table** with real-time status updates for each file
- ⏱️ **Elapsed timer** shows processing time for current file
- 🔢 Natural sorting (NeueAufnahme1, NeueAufnahme2, ..., NeueAufnahme10)
- 📝 Combined output with file separators
- 📈 Summary with **speed metric** (e.g., "1.9x realtime")

**Supported formats:**
`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`

### Command help

```bash
python whisp.py --help
```

## 📊 Example Output

```
╭─────────────────────────────────────────╮
│ Whisper                                 │
│ Powered by OpenAI large-v3              │
╰─────────────────────────────────────────╯

Device: cpu

Initializing Whisper model...
Model: medium (Good balance, ~1.5GB)
Using faster-whisper for optimized performance
Downloading model 'Systran/faster-whisper-medium'...
Downloading model... ████████████████████████████████████████ 100% • 1.5/1.5 GB • 5.2 MB/s • 00:00:00
✓ Model downloaded successfully
Loading model with compute type 'int8'...
✓ Model loaded successfully on cpu
Compute type: int8

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

### Batch Mode Output Example

```
📦 BATCH MODE
Directory: ./lectures/
Output: transcript.txt

Found 5 audio files:
Total duration: 42:15

Processing...
╭──────┬──────────────────────┬──────────┬──────────────────╮
│    # │ File                 │ Duration │ Status           │
├──────┼──────────────────────┼──────────┼──────────────────┤
│    1 │ Neue Aufnahme 01.m4a │    08:03 │ ✓ 1247 words     │
│    2 │ Neue Aufnahme 02.m4a │    08:10 │ ✓ 1156 words     │
│    3 │ Neue Aufnahme 03.m4a │    07:08 │ ⏹ 3:42           │
│    4 │ Neue Aufnahme 04.m4a │    08:31 │                  │
│    5 │ Neue Aufnahme 05.m4a │    10:23 │                  │
╰──────┴──────────────────────┴──────────┴──────────────────╯

Summary:
╭──────────────────┬───────────────╮
│ Files processed  │ 5             │
│ Total duration   │ 42:15         │
│ Processing time  │ 22:30         │
│ Speed            │ 1.9x realtime │
│ Total words      │ 6,543         │
│ Total characters │ 41,234        │
╰──────────────────┴───────────────╯

Batch transcription completed successfully!
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

- **large** - latest v3 model, use for academic lectures, medical recordings, technical documentation (any language)
- **large-v2** - previous large version, slightly faster than v3, similar accuracy
- **turbo** - optimal for fast transcription with good accuracy (any language)
- **medium** - good balance for any language on moderate hardware
- **small** - for simple recordings with good audio quality
- **base** - quick tests or very simple content

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

### Interrupted model download

If your model download was interrupted, simply run the script again. Downloads are resumable — only missing files will be downloaded.

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
