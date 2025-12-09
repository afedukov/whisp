# 🎙️ Whisper

Fast and accurate audio transcription CLI powered by [OpenAI Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

**Features:**
- 🎯 Transcribe audio files with high accuracy
- 🎙️ Record from microphone and transcribe in one command
- 🗂️ Batch process entire folders
- 🌐 **Auto-translate** transcriptions to any language (powered by OpenAI GPT)
- ⚡ **4-8x faster** than standard Whisper implementations
- 🌍 99+ languages supported with auto-detection

## 🚀 Quick Start

```bash
# Transcribe a single file
whisp audio.mp3                          # → audio.txt

# Record and transcribe
whisp record                             # → ~/Records/recording_TIMESTAMP.txt

# Batch process folder
whisp ./recordings/                      # → recordings.txt

# With specific model and language
whisp audio.mp3 --model large --language de

# With translation to target language
whisp audio.mp3 --translate ru             # → audio.txt + audio_ru.txt
```

> 💡 **Note:** If you installed with `pip install -r requirements.txt` instead of `pip install -e .`, use `python whisp.py` instead of `whisp`

## ✨ Full Feature List

- 🎯 High accuracy transcription with Whisper Large-v3
- ⚡ 4-8x faster than standard Whisper (using CTranslate2)
- 🚀 GPU (CUDA) support for accelerated processing
- 💾 Lower memory usage with int8 quantization on CPU
- 🌍 Automatic language detection or manual language specification
- 📝 Preview of transcription results
- 🔄 Multiple model options (large, large-v2, turbo, medium, small, base)
- 🎤 Voice activity detection (VAD) to skip silence
- 🎙️ Microphone recording mode - record and transcribe
- 🗂️ Batch mode - process entire folders
- 💾 M4A compression - save recordings 10x smaller with minimal quality loss
- 🌐 **Auto-translation** - translate transcriptions to any language using OpenAI GPT API

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

### 4. Install the package

**Option A: Install as editable package (recommended)**
```bash
pip install --upgrade pip
pip install -e .
```

**Option B: Install from requirements.txt (use `python whisp.py`)**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note:** The first run will take some time as the Whisper model (~3GB for large) will be downloaded. The model is cached locally for future use. Downloads can be interrupted and resumed.

### 5. Using the `whisp` command

After installing with **Option A** (`pip install -e .`), you can use the `whisp` command from anywhere:

```bash
whisp audio.mp3              # Instead of: python whisp.py audio.mp3
whisp record                 # Instead of: python whisp.py record
whisp ./recordings/          # Instead of: python whisp.py ./recordings/
```

If you used **Option B**, use `python whisp.py` instead.

### Creating a permanent alias (Optional)

To run `whisp` from any directory without manually activating the virtual environment every time, you can add an alias to your shell configuration (e.g., `~/.zshrc`).

Run this command from the project root directory:

```bash
echo "alias whisp=\"$(pwd)/venv/bin/whisp\"" >> ~/.zshrc
source ~/.zshrc
```


## 💻 Usage

### Single File Transcription

**Basic usage:**
```bash
whisp input.mp3 output.txt
whisp input.mp3                    # Auto-generates input.txt
```

**With model and language specification:**
```bash
whisp audio.wav transcript.txt --model large --language en
whisp audio.wav --model turbo      # Auto-generates audio.txt
```

**Example commands:**
```bash
# German lecture with maximum accuracy
whisp lecture.mp3 --model large --language de

# Fast podcast transcription
whisp podcast.m4a --model turbo --language en

# Auto-detect language with medium model
whisp interview.mp3 --model medium
```

### Model Selection

Available models:

| Model | Size | Accuracy | Speed | Recommendation |
|-------|------|----------|-------|----------------|
| **turbo** | ~800MB | Good | 8x faster | ✅ Default, fast multilingual |
| **large** | ~3GB | Best | Slow | For academic content (latest v3) |
| **large-v2** | ~3GB | Best | Slightly faster | Previous large version |
| **medium** | ~1.5GB | Good | 2-3x faster | Balance of speed and quality |
| **small** | ~466MB | Basic | Fast | For simple tasks |
| **base** | ~145MB | Basic | Very fast | Minimal accuracy |

### Recording Mode (Microphone Input)

Record audio from your microphone and transcribe it automatically.

**Basic recording:**
```bash
whisp record output.txt
whisp record                          # Auto-saves to save_dir/recording_TIMESTAMP.txt
```

**Recording with specific model and language:**
```bash
whisp record transcript.txt --model turbo --language de
whisp record --model turbo --language de  # Auto-generates filename
```

**How it works:**
1. Shows list of available microphones (or uses `default_device` from config)
2. You select a device with arrow keys ↑↓ and Enter
3. Press Enter to start recording
4. Press **Ctrl+D** to stop recording (prevents accidental stops)
5. Audio is automatically transcribed using selected model
6. Both audio and transcription saved to `save_dir` with matching timestamps
7. Files named like: `recording_20251208_195410.m4a` and `recording_20251208_195410.txt`

**Permissions on macOS:**
- First run will ask for microphone permission
- If denied: System Settings → Privacy & Security → Microphone → Terminal



### Batch Mode (Directory Input)

Process entire folders of audio files. All files are processed in **natural sort order** (1, 2, 10 instead of 1, 10, 2) and combined into a single output file.

**Basic batch processing:**
```bash
whisp ./lectures/ combined_transcript.txt --language de --model turbo
whisp ./recordings/                    # Auto-generates recordings.txt
```

**Supported formats:**
`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`

### Translation Mode

Automatically translate transcriptions to any target language using OpenAI GPT API.

**Basic translation:**
```bash
whisp audio.mp3 --translate ru              # → audio.txt + audio_ru.txt
whisp audio.mp3 --translate en --language de  # German audio → English translation
```

**With recording mode:**
```bash
whisp record --translate ru
# Creates: recording_20251209_072149.m4a + recording_20251209_072149.txt + recording_20251209_072149_ru.txt
```

**With batch mode:**
```bash
whisp ./lectures/ --language de --translate ru
# Creates: lectures.txt + lectures_ru.txt (all files combined)
```

**Setup:**
1. Get OpenAI API key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Add to `config.yaml`:
   ```yaml
   translation:
     openai_api_key: "sk-..."
     model: "gpt-5-mini"  # Recommended: fast, excellent quality
   ```
   Or set environment variable: `export OPENAI_API_KEY="sk-..."`

**Features:**
- High-quality contextual translation with GPT
- Automatic paragraph organization for readability
- Low cost: ~$0.05 per 1.5-hour lecture (gpt-5-mini)
- Preserves technical terms, names, and numbers
- Customizable translation prompt in config.yaml
- Supports all languages (en, ru, de, es, fr, ja, zh, etc.)

**Cost estimate (gpt-5-mini):**
- Short audio (5 min): ~$0.003
- Medium audio (30 min): ~$0.01
- Long lecture (1.5 hours): ~$0.05

> 💡 **Tip:** Translation is optional. If `--translate` is not specified, only transcription is saved.

### Command Help

```bash
python whisp.py --help
```

## ⚙️ Configuration

The application can be configured via `config.yaml` file. All settings have sensible defaults.
    
    **Configuration Loading Order (Priority High to Low):**
    1. `~/.whisp/config.yaml` in user home directory (User global)
    2. `config.yaml` in application directory (Default)

### Transcription Settings
- `default_language`: Auto-detect if empty, or specify (e.g., "en", "de", "ru")
- `beam_size`: Search beam size (default: 5) - higher = more accurate but slower
- `vad_filter`: Voice activity detection to skip silence (default: true)
- `min_silence_duration_ms`: Minimum silence duration for VAD (default: 500ms)

### Recording Settings
- `sample_rate`: Recording quality (default: 16000 Hz)
- `channels`: Audio channels (default: 1 - mono)
- `default_device`: Pre-select microphone by name or show menu with `-1`
- `save_dir`: Directory for saved recordings
- `keep_recording`: Keep audio file after transcription (default: false)
- `compress_format`: `"m4a"` (10x smaller) or `"wav"` (default: m4a)
- `show_level_meter`: Show real-time audio level (default: true)

### Model Settings
- `default`: Model to use if not specified (default: "turbo")
- `compute_type_cpu`: Quantization for CPU (default: "int8")
- `compute_type_gpu`: Precision for GPU (default: "float16")

### Output Settings
- `preview_length`: Characters to show in preview (default: 200)

### Translation Settings
- `openai_api_key`: Your OpenAI API key (get at [platform.openai.com/api-keys](https://platform.openai.com/api-keys))
- `model`: GPT model for translation (default: "gpt-5-mini")
  - Options: gpt-5-mini (recommended), gpt-5-nano, gpt-4o-mini
- `temperature`: Creativity (default: 1.0 for gpt-5-mini, 0.3 otherwise)
- `system_prompt`: AI translator's role/persona
- `user_prompt`: Specific translation instructions

## 🎯 Model Selection Guide

### Recommended Models

- **turbo** - ✅ Best for most use cases: fast (8x) with good accuracy
- **large** - Maximum accuracy for academic/technical content (latest v3)
- **large-v2** - Previous version, slightly faster than v3
- **medium** - Good balance for any language
- **small** - Fast transcription with acceptable quality
- **base** - Quick tests only

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

If transcription is slower than expected:

1. Check that `faster-whisper` is properly installed:
```bash
python -c "from faster_whisper import WhisperModel; print('OK')"
```

2. Verify you're using CPU int8 quantization (check console output)
3. Try a smaller model (turbo or medium) for faster processing
4. Ensure VAD filter is enabled in config.yaml (skips silence)

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
├── pyproject.toml          # Project metadata and install config
├── requirements.txt        # Python dependencies
├── config.yaml            # Configuration file
├── README.md              # Documentation
└── venv/                  # Virtual environment

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
