#!/usr/bin/env python3
"""
Whisper Audio Transcription Tool
Transcribe audio files using OpenAI's Whisper large-v3 model
"""

import argparse
import sys
import os
import warnings
import tempfile
import subprocess
import shutil
from pathlib import Path

# Enable download progress bars for huggingface_hub
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable hf_transfer to use tqdm progress

try:
    from huggingface_hub import logging as hf_logging
    hf_logging.set_verbosity_info()
except ImportError:
    pass

from faster_whisper import WhisperModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import box

# Filter out warnings but allow tqdm progress bars
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# Available Whisper models (faster-whisper uses model size names directly)
WHISPER_MODELS = {
    "large": "large-v3",
    "distil": "distil-large-v3",  # Distilled version - faster with good accuracy (English-optimized)
    "medium": "medium",
    "small": "small",
    "base": "base"
}

MODEL_INFO = {
    "large": "Best accuracy, ~3GB (recommended for academic content)",
    "distil": "Distilled large-v3, ~1.5GB, 6x faster (English-optimized)",
    "medium": "Good balance, ~1.5GB",
    "small": "Fast, ~466MB, lower accuracy",
    "base": "Very fast, ~145MB, basic accuracy"
}


def print_header(model_size: str = "large"):
    """Print a beautiful header for the application"""
    header = Text()
    header.append("Whisper\n", style="bold cyan")
    model_name = WHISPER_MODELS.get(model_size, WHISPER_MODELS["large"]).split("/")[-1]
    header.append(f"Powered by OpenAI {model_name}", style="dim")

    console.print(Panel(
        header,
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2)
    ))


def check_file_exists(file_path: Path) -> bool:
    """Check if input file exists"""
    if not file_path.exists():
        console.print(f"[bold red]✗[/bold red] Error: Input file not found: {file_path}")
        return False
    return True


def get_audio_duration(audio_file: Path) -> float:
    """Get audio duration in seconds using ffprobe"""
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_file)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            return 0.0
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS or HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def convert_audio_to_wav(audio_file: Path) -> Path:
    """Convert audio to WAV format using ffmpeg if needed"""
    # Check if file is already in a format that soundfile can handle
    supported_formats = ['.wav', '.flac']
    if audio_file.suffix.lower() in supported_formats:
        return audio_file

    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        console.print("[bold red]✗[/bold red] ffmpeg not found in PATH")
        console.print("[yellow]Please install ffmpeg to convert audio files[/yellow]")
        console.print("[dim]macOS: brew install ffmpeg[/dim]")
        console.print("[dim]Ubuntu: sudo apt install ffmpeg[/dim]")
        sys.exit(1)

    console.print(f"[cyan]Converting {audio_file.suffix} to WAV format...[/cyan]")

    try:
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = Path(temp_wav.name)
        temp_wav.close()

        # Convert using ffmpeg
        # -i: input file
        # -ar 16000: sample rate 16kHz (good for speech)
        # -ac 1: mono audio
        # -y: overwrite output file
        result = subprocess.run(
            [
                'ffmpeg',
                '-i', str(audio_file),
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(temp_wav_path)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            console.print(f"[bold red]✗[/bold red] FFmpeg conversion failed")
            console.print(f"[dim]{result.stderr}[/dim]")
            raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

        console.print("[green]✓[/green] Audio converted successfully")
        return temp_wav_path

    except FileNotFoundError:
        console.print("[bold red]✗[/bold red] ffmpeg not found")
        console.print("[yellow]Please install ffmpeg to convert audio files[/yellow]")
        raise
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Audio conversion failed: {str(e)}")
        raise


def download_model_with_progress(model_id: str) -> str:
    """Download model from HuggingFace Hub with progress bar, return local path.
    
    Uses snapshot_download which automatically:
    - Resumes interrupted downloads
    - Verifies file integrity via checksums
    - Uses cached files if already downloaded
    """
    from huggingface_hub import snapshot_download
    
    # Map faster-whisper model names to HuggingFace repo names
    repo_map = {
        "large-v3": "Systran/faster-whisper-large-v3",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
        "medium": "Systran/faster-whisper-medium",
        "small": "Systran/faster-whisper-small",
        "base": "Systran/faster-whisper-base",
    }
    
    repo_id = repo_map.get(model_id, f"Systran/faster-whisper-{model_id}")
    
    console.print(f"[cyan]Checking model '{repo_id}'...[/cyan]")
    
    # Download with progress (will use cache if complete, resume if partial)
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_files_only=False,
        )
        console.print(f"[green]✓[/green] Model ready")
        return local_path
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to download model: {str(e)}")
        raise


def load_model(device: str, model_size: str = "large"):
    """Load Whisper model using faster-whisper"""
    model_id = WHISPER_MODELS.get(model_size, WHISPER_MODELS["large"])
    model_info = MODEL_INFO.get(model_size, MODEL_INFO["large"])

    console.print("\n[bold cyan]Initializing Whisper model...[/bold cyan]")
    console.print(f"[dim]Model: {model_size} ({model_info})[/dim]")
    console.print("[dim]Using faster-whisper for optimized performance[/dim]")

    try:
        # Determine compute type based on device
        # int8 is good balance of speed and accuracy for CPU
        # float16 for GPU (if available)
        if device == "cuda" or device.startswith("cuda:"):
            compute_type = "float16"
            device_type = "cuda"
        else:
            compute_type = "int8"
            device_type = "cpu"

        # First download model with progress bar
        model_path = download_model_with_progress(model_id)

        console.print(f"[dim]Loading model with compute type '{compute_type}'...[/dim]")

        # Load model with faster-whisper from local path
        model = WhisperModel(
            model_path,
            device=device_type,
            compute_type=compute_type,
            local_files_only=True  # Use already downloaded model
        )

        console.print(f"[bold green]✓[/bold green] Model loaded successfully on [bold]{device_type}[/bold]")
        console.print(f"[dim]Compute type: {compute_type}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]! Model loading cancelled by user[/yellow]")
        raise  # Re-raise to be caught by main()
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Failed to load model: {str(e)}")
        raise

    return model


def transcribe_audio(audio_file: Path, output_file: Path, language: str = None, model_size: str = "large"):
    """Transcribe audio file to text"""

    print_header(model_size)

    # Check if input file exists
    if not check_file_exists(audio_file):
        sys.exit(1)

    # Determine device (faster-whisper handles this internally)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"\n[dim]Device: {device}[/dim]")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"[dim]GPU: {gpu_name}[/dim]")
    except ImportError:
        # torch not installed, use CPU
        device = "cpu"
        console.print(f"\n[dim]Device: {device}[/dim]")

    # Load model
    try:
        model = load_model(device, model_size)
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Failed to load model: {str(e)}")
        sys.exit(1)

    # Convert audio if needed
    temp_file = None
    try:
        converted_audio = convert_audio_to_wav(audio_file)
        if converted_audio != audio_file:
            temp_file = converted_audio  # Mark for cleanup later

        # Get audio duration
        audio_duration = get_audio_duration(audio_file)
        duration_str = format_duration(audio_duration) if audio_duration > 0 else "unknown"

        # Transcribe
        console.print(f"\n[bold cyan]Transcribing audio file...[/bold cyan]")
        console.print(f"[dim]Input: {audio_file}[/dim]")
        console.print(f"[dim]Duration: {duration_str}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Processing {duration_str} of audio...",
                total=None
            )

            # Transcribe with faster-whisper
            segments, info = model.transcribe(
                str(converted_audio),
                language=language,
                task="transcribe",
                beam_size=5,
                vad_filter=True,  # Voice activity detection to skip silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect all segments into full transcription
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text)

            transcription = " ".join(transcription_parts).strip()

            progress.update(task, completed=True)

        # Log detected language
        if info.language_probability > 0:
            console.print(f"[dim]Detected language: {info.language} (confidence: {info.language_probability:.2%})[/dim]")

        # Save to file
        console.print(f"\n[bold cyan]Saving transcription...[/bold cyan]")
        output_file.write_text(transcription, encoding='utf-8')

        console.print(f"[bold green]✓[/bold green] Transcription saved to: {output_file}")

        # Display preview
        preview_length = 200
        preview = transcription[:preview_length]
        if len(transcription) > preview_length:
            preview += "..."

        console.print("\n[bold cyan]Preview:[/bold cyan]")
        console.print(Panel(
            preview,
            box=box.ROUNDED,
            border_style="green",
            padding=(1, 2)
        ))

        # Stats
        word_count = len(transcription.split())
        char_count = len(transcription)
        console.print(f"\n[dim]Stats: {word_count} words, {char_count} characters[/dim]")

        console.print("\n[bold green]Transcription completed successfully![/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]! Transcription cancelled by user[/yellow]")
        console.print("[dim]Cleaning up...[/dim]\n")
        raise  # Re-raise to be caught by main()
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Transcription failed: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temporary file if created
        if temp_file and temp_file.exists():
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp3 output.txt
  %(prog)s audio.wav transcript.txt --language en
  %(prog)s lecture.mp3 transcript.txt --language de --model large
  %(prog)s podcast.m4a transcript.txt --model medium --language ru

Available models:
  large  - Best accuracy, ~3GB (default, recommended for academic content)
  distil - Distilled large-v3, ~1.5GB, 6x faster (English-optimized)
  medium - Good balance, ~1.5GB, 2-3x faster than large
  small  - Fast, ~466MB, lower accuracy
  base   - Very fast, ~145MB, basic accuracy
        """
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Input audio file (mp3, wav, m4a, etc.)"
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Output text file for transcription"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["large", "distil", "medium", "small", "base"],
        default="large",
        help="Whisper model size (default: large)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'ru', 'de', 'es'). If not specified, will auto-detect."
    )

    args = parser.parse_args()

    try:
        transcribe_audio(args.input_file, args.output_file, args.language, args.model)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]! Operation cancelled by user[/yellow]")
        console.print("[dim]Exiting gracefully...[/dim]\n")
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
