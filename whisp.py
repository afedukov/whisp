#!/usr/bin/env python3
"""
Whisper Audio Transcription Tool
Transcribe audio files using OpenAI's Whisper large-v3 model
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
import warnings
import tempfile
import os
import subprocess
import shutil

# Allow tqdm to show (remove filtering)
warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# Available Whisper models
WHISPER_MODELS = {
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small"
}

MODEL_INFO = {
    "large": "Best accuracy, ~3GB, slower (recommended for academic content)",
    "turbo": "Same accuracy as large, 8x faster, ~1.5GB",
    "medium": "Good balance, ~1.5GB, 2-3x faster than large",
    "small": "Fast, ~466MB, lower accuracy"
}


def print_header(model_size: str = "large"):
    """Print a beautiful header for the application"""
    header = Text()
    header.append("🎙️  Whisper Transcription Tool\n", style="bold cyan")
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


def load_model(device: str, model_size: str = "large"):
    """Load Whisper model and processor"""
    model_id = WHISPER_MODELS.get(model_size, WHISPER_MODELS["large"])
    model_info = MODEL_INFO.get(model_size, MODEL_INFO["large"])

    console.print("\n[bold cyan]⚙️  Initializing Whisper model...[/bold cyan]")
    console.print(f"[dim]Model: {model_size} ({model_info})[/dim]")
    console.print("[dim]Downloading model files (this may take a few minutes on first run)...[/dim]\n")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        # Load model - HuggingFace will show native tqdm progress bars
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        console.print("\n[cyan]Loading model to device...[/cyan]")
        model.to(device)

        console.print("[cyan]Loading processor...[/cyan]")
        processor = AutoProcessor.from_pretrained(model_id)

        console.print(f"[bold green]✓[/bold green] Model loaded successfully on [bold]{device}[/bold]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Model loading cancelled by user[/yellow]")
        raise  # Re-raise to be caught by main()

    return model, processor


def transcribe_audio(audio_file: Path, output_file: Path, language: str = None, model_size: str = "large"):
    """Transcribe audio file to text"""

    print_header(model_size)

    # Check if input file exists
    if not check_file_exists(audio_file):
        sys.exit(1)

    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[dim]Device: {device}[/dim]")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[dim]GPU: {gpu_name}[/dim]")

    # Load model
    try:
        model, processor = load_model(device, model_size)
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Failed to load model: {str(e)}")
        sys.exit(1)

    # Create pipeline
    console.print("\n[bold cyan]🔧 Creating transcription pipeline...[/bold cyan]")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device,
    )

    console.print("[bold green]✓[/bold green] Pipeline ready")

    # Convert audio if needed
    temp_file = None
    try:
        converted_audio = convert_audio_to_wav(audio_file)
        if converted_audio != audio_file:
            temp_file = converted_audio  # Mark for cleanup later

        # Transcribe
        console.print(f"\n[bold cyan]🎵 Transcribing audio file...[/bold cyan]")
        console.print(f"[dim]Input: {audio_file}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing audio...", total=None)

            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language

            result = pipe(str(converted_audio), generate_kwargs=generate_kwargs)

            progress.update(task, completed=True)

        transcription = result["text"]

        # Save to file
        console.print(f"\n[bold cyan]💾 Saving transcription...[/bold cyan]")
        output_file.write_text(transcription, encoding='utf-8')

        console.print(f"[bold green]✓[/bold green] Transcription saved to: {output_file}")

        # Display preview
        preview_length = 200
        preview = transcription[:preview_length]
        if len(transcription) > preview_length:
            preview += "..."

        console.print("\n[bold cyan]📝 Preview:[/bold cyan]")
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

        console.print("\n[bold green]✨ Transcription completed successfully![/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Transcription cancelled by user[/yellow]")
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
            except:
                pass  # Ignore cleanup errors


def main():
    parser = argparse.ArgumentParser(
        description="🎙️  Transcribe audio files using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp3 output.txt
  %(prog)s audio.wav transcript.txt --language en
  %(prog)s lecture.mp3 transcript.txt --language de --model large
  %(prog)s podcast.m4a transcript.txt --model turbo --language ru

Available models:
  large  - Best accuracy, ~3GB (default, recommended for academic content)
  turbo  - Same accuracy as large, 8x faster, ~1.5GB
  medium - Good balance, ~1.5GB, 2-3x faster than large
  small  - Fast, ~466MB, lower accuracy
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
        choices=["large", "turbo", "medium", "small"],
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
        console.print("\n\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        console.print("[dim]Exiting gracefully...[/dim]\n")
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        console.print("[dim]Exiting gracefully...[/dim]\n")
        sys.exit(130)
