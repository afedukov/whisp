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
    "turbo": "large-v3-turbo",  # 8x faster than large-v3, multilingual (de, en, ru, etc.)
    "medium": "medium",
    "small": "small",
    "base": "base"
}

MODEL_INFO = {
    "large": "Best accuracy, ~3GB",
    "turbo": "Large-v3-turbo, ~800MB, 8x faster, multilingual",
    "medium": "Good balance, ~1.5GB",
    "small": "Fast, ~466MB, lower accuracy",
    "base": "Very fast, ~145MB, basic accuracy"
}

# Default configuration values
DEFAULT_CONFIG = {
    "model": {
        "default": "turbo",
        "compute_type_cpu": "int8",
        "compute_type_gpu": "float16"
    },
    "transcription": {
        "default_language": "",
        "beam_size": 5,
        "vad_filter": True,
        "min_silence_duration_ms": 500
    },
    "output": {
        "preview_length": 200
    }
}

def load_config() -> dict:
    """Load configuration from config.yaml in the app directory.
    Falls back to default values if file not found or invalid.
    """
    import yaml
    
    config_path = Path(__file__).parent / "config.yaml"
    config = DEFAULT_CONFIG.copy()
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            
            if user_config:
                # Deep merge user config with defaults
                for section in DEFAULT_CONFIG:
                    if section in user_config and isinstance(user_config[section], dict):
                        config[section] = {**DEFAULT_CONFIG[section], **user_config[section]}
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config.yaml: {e}[/yellow]")
            console.print("[dim]Using default configuration[/dim]")
    
    return config

# Load configuration at startup
CONFIG = load_config()


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


def convert_audio_to_wav(audio_file: Path, quiet: bool = False) -> Path:
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

    if not quiet:
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

        if not quiet:
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
    """Download model from HuggingFace Hub with beautiful progress bar.
    
    Uses hf_hub_download for each file with rich progress tracking.
    Automatically resumes interrupted downloads and verifies integrity.
    """
    from huggingface_hub import hf_hub_download, list_repo_files, try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE
    from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TaskID
    import threading
    
    # Map faster-whisper model names to HuggingFace repo names
    repo_map = {
        "large-v3": "Systran/faster-whisper-large-v3",
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "medium": "Systran/faster-whisper-medium",
        "small": "Systran/faster-whisper-small",
        "base": "Systran/faster-whisper-base",
    }
    
    repo_id = repo_map.get(model_id, f"Systran/faster-whisper-{model_id}")
    
    # Check if model is already fully cached
    cache_dir = Path(HF_HUB_CACHE)
    repo_folder = cache_dir / f"models--{repo_id.replace('/', '--')}"
    
    try:
        # Get list of files in the repo
        files = list_repo_files(repo_id)
        
        # Check if all files are cached
        all_cached = True
        for filename in files:
            cached = try_to_load_from_cache(repo_id, filename)
            if cached is None:
                all_cached = False
                break
        
        if all_cached:
            console.print(f"[green]✓[/green] Model '{model_id}' found in cache")
            # Return the snapshot path
            snapshots_dir = repo_folder / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    return str(snapshots[0])
    except Exception:
        pass  # Continue with download
    
    console.print(f"[cyan]Downloading model '{repo_id}'...[/cyan]")
    
    # Custom progress bar in app style
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        
        # Get file list and sizes
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_info = api.repo_info(repo_id, files_metadata=True)
            
            # Calculate total size
            file_sizes = {}
            total_size = 0
            for sibling in repo_info.siblings:
                if sibling.size:
                    file_sizes[sibling.rfilename] = sibling.size
                    total_size += sibling.size
            
            # Create main progress task
            main_task = progress.add_task(
                f"[cyan]Downloading model...",
                total=total_size
            )
            
            downloaded_size = 0
            local_path = None
            
            # Download each file
            for filename in file_sizes:
                file_size = file_sizes[filename]
                
                # Check if already cached
                cached = try_to_load_from_cache(repo_id, filename)
                if cached:
                    downloaded_size += file_size
                    progress.update(main_task, completed=downloaded_size)
                    continue
                
                # Download with progress callback
                def progress_callback(current: int, total: int):
                    progress.update(main_task, completed=downloaded_size + current)
                
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    # Note: hf_hub_download doesn't have a progress callback,
                    # but it shows tqdm which we've configured to use rich
                )
                
                downloaded_size += file_size
                progress.update(main_task, completed=downloaded_size)
            
            progress.update(main_task, completed=total_size)
            
        except Exception as e:
            progress.stop()
            console.print(f"[yellow]! Could not get file sizes, using simple download...[/yellow]")
            # Fallback to simple download
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(repo_id=repo_id, local_files_only=False)
    
    # Get the snapshot path
    snapshots_dir = repo_folder / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            console.print(f"[green]✓[/green] Model downloaded successfully")
            return str(snapshots[0])
    
    raise RuntimeError(f"Failed to find downloaded model in cache: {repo_folder}")


def load_model(device: str, model_size: str = "large"):
    """Load Whisper model using faster-whisper"""
    model_id = WHISPER_MODELS.get(model_size, WHISPER_MODELS["large"])
    model_info = MODEL_INFO.get(model_size, MODEL_INFO["large"])

    console.print("\n[bold cyan]Initializing Whisper model...[/bold cyan]")
    console.print(f"[dim]Model: {model_size} ({model_info})[/dim]")
    console.print("[dim]Using faster-whisper for optimized performance[/dim]")

    try:
        # Determine compute type based on device (from config)
        if device == "cuda" or device.startswith("cuda:"):
            compute_type = CONFIG["model"]["compute_type_gpu"]
            device_type = "cuda"
        else:
            compute_type = CONFIG["model"]["compute_type_cpu"]
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

    try:
        import time
        
        # Get audio duration
        audio_duration = get_audio_duration(audio_file)
        duration_str = format_duration(audio_duration) if audio_duration > 0 else "unknown"

        # Transcribe
        console.print(f"\n[bold cyan]Transcribing audio file...[/bold cyan]")
        console.print(f"[dim]Input: {audio_file}[/dim]")
        console.print(f"[dim]Duration: {duration_str}[/dim]")
        
        transcribe_start_time = time.time()

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

            # Transcribe with faster-whisper (handles all formats via ffmpeg)
            segments, info = model.transcribe(
                str(audio_file),
                language=language,
                task="transcribe",
                beam_size=CONFIG["transcription"]["beam_size"],
                vad_filter=CONFIG["transcription"]["vad_filter"],
                vad_parameters=dict(min_silence_duration_ms=CONFIG["transcription"]["min_silence_duration_ms"])
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
        preview_length = CONFIG["output"]["preview_length"]
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
        from rich.table import Table
        
        processing_time = time.time() - transcribe_start_time
        speed_ratio = audio_duration / processing_time if processing_time > 0 and audio_duration > 0 else 0
        
        word_count = len(transcription.split())
        char_count = len(transcription)
        
        console.print("\n[bold cyan]Stats:[/bold cyan]")
        stats_table = Table(box=box.ROUNDED, border_style="green", show_header=False)
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")
        stats_table.add_row("Duration", duration_str)
        stats_table.add_row("Processing time", format_duration(processing_time))
        stats_table.add_row("Speed", f"{speed_ratio:.1f}x realtime")
        stats_table.add_row("Words", f"{word_count:,}")
        stats_table.add_row("Characters", f"{char_count:,}")
        console.print(stats_table)

        console.print("\n[bold green]Transcription completed successfully![/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]! Transcription cancelled by user[/yellow]")
        console.print("[dim]Cleaning up...[/dim]\n")
        raise  # Re-raise to be caught by main()
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Transcription failed: {str(e)}")
        sys.exit(1)


# =============================================================================
# Batch Mode Functions
# =============================================================================

import re

# Supported audio extensions for batch mode
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}


def natural_sort_key(path: Path) -> list:
    """Generate a key for natural sorting (1, 2, 10 instead of 1, 10, 2)"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', path.name)]


def get_audio_files(directory: Path) -> list[Path]:
    """Get all audio files from directory, sorted naturally"""
    audio_files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file)
    
    # Sort naturally (1, 2, 10 instead of 1, 10, 2)
    audio_files.sort(key=natural_sort_key)
    return audio_files


def transcribe_single_file(model, audio_file: Path, language: str = None, quiet: bool = False) -> tuple[str, dict]:
    """Transcribe a single audio file using pre-loaded model.
    Returns (transcription_text, info_dict)
    
    Args:
        quiet: If True, suppress console output (for batch mode)
    
    Note: faster-whisper handles audio formats (m4a, mp3, etc.) directly via ffmpeg
    """
    # Transcribe with faster-whisper (handles all formats via ffmpeg)
    segments, info = model.transcribe(
        str(audio_file),
        language=language,
        task="transcribe",
        beam_size=CONFIG["transcription"]["beam_size"],
        vad_filter=CONFIG["transcription"]["vad_filter"],
        vad_parameters=dict(min_silence_duration_ms=CONFIG["transcription"]["min_silence_duration_ms"])
    )

    # Collect all segments
    transcription_parts = []
    for segment in segments:
        transcription_parts.append(segment.text)

    transcription = " ".join(transcription_parts).strip()
    
    info_dict = {
        "language": info.language,
        "probability": info.language_probability
    }
    
    return transcription, info_dict


def transcribe_batch(input_dir: Path, output_file: Path, language: str = None, model_size: str = "large"):
    """Transcribe all audio files in a directory (batch mode)"""
    from rich.table import Table
    from rich.live import Live
    
    print_header(model_size)
    
    # Check directory exists
    if not input_dir.exists():
        console.print(f"[bold red]✗[/bold red] Error: Directory not found: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        console.print(f"[bold red]✗[/bold red] Error: Not a directory: {input_dir}")
        sys.exit(1)
    
    # Get audio files
    audio_files = get_audio_files(input_dir)
    
    if not audio_files:
        console.print(f"[bold red]✗[/bold red] No audio files found in: {input_dir}")
        console.print(f"[dim]Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}[/dim]")
        sys.exit(1)
    
    # Print batch mode header
    console.print("\n[bold magenta]📦 BATCH MODE[/bold magenta]")
    console.print(f"[dim]Directory: {input_dir}[/dim]")
    console.print(f"[dim]Output: {output_file}[/dim]")
    
    # Calculate total duration
    total_duration = 0.0
    file_durations = []
    for f in audio_files:
        dur = get_audio_duration(f)
        file_durations.append(dur)
        total_duration += dur
    
    console.print(f"\n[bold cyan]Found {len(audio_files)} audio files:[/bold cyan]")
    console.print(f"[dim]Total duration: {format_duration(total_duration)}[/dim]\n")
    
    # Determine device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[dim]Device: {device}[/dim]")
    except ImportError:
        device = "cpu"
        console.print(f"[dim]Device: {device}[/dim]")
    
    # Load model once
    try:
        model = load_model(device, model_size)
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Initialize status tracking
    # Status: "" = pending, "⏹ MM:SS" = in progress with elapsed time, "✓ N words" = done, "✗ error" = failed
    file_statuses = ["" for _ in audio_files]
    current_file_index = [-1]  # Use list to allow mutation from inner function
    processing_start_time = [0.0]
    
    import time
    import threading
    
    def make_table():
        """Create the progress table with current statuses"""
        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Duration", style="green", justify="right", width=8)
        table.add_column("Status", justify="left", width=16)
        
        for i, (f, dur, status) in enumerate(zip(audio_files, file_durations, file_statuses), 1):
            dur_str = format_duration(dur) if dur > 0 else "?"
            # If this is the currently processing file, show elapsed time
            if i - 1 == current_file_index[0] and processing_start_time[0] > 0:
                elapsed = time.time() - processing_start_time[0]
                elapsed_str = format_duration(elapsed)
                status = f"[yellow]⏹ {elapsed_str}[/yellow]"
            table.add_row(str(i), f.name, dur_str, status)
        
        return table
    
    # Process files with live table
    console.print(f"\n[bold cyan]Processing...[/bold cyan]")
    
    all_transcriptions = []
    total_words = 0
    total_chars = 0
    batch_start_time = time.time()  # Track total processing time
    
    stop_refresh = threading.Event()
    
    def refresh_loop(live):
        """Background thread to refresh elapsed time"""
        while not stop_refresh.is_set():
            live.update(make_table())
            time.sleep(0.5)
    
    with Live(make_table(), console=console, refresh_per_second=4) as live:
        # Start background refresh thread
        refresh_thread = threading.Thread(target=refresh_loop, args=(live,), daemon=True)
        refresh_thread.start()
        
        try:
            for i, audio_file in enumerate(audio_files):
                # Update status to "processing"
                current_file_index[0] = i
                processing_start_time[0] = time.time()
                live.update(make_table())
                
                try:
                    transcription, info = transcribe_single_file(model, audio_file, language, quiet=True)
                    
                    # Just add text with newline separator (no file headers)
                    all_transcriptions.append(transcription)
                    
                    word_count = len(transcription.split())
                    char_count = len(transcription)
                    total_words += word_count
                    total_chars += char_count
                    
                    # Calculate elapsed time for this file
                    elapsed = time.time() - processing_start_time[0]
                    elapsed_str = format_duration(elapsed)
                    
                    # Update status to "done" with time
                    file_statuses[i] = f"[green]✓ {word_count} words[/green]"
                    
                except Exception as e:
                    file_statuses[i] = f"[red]✗ error[/red]"
                    all_transcriptions.append(f"[ERROR: {audio_file.name}: {str(e)}]")
                
                current_file_index[0] = -1
                live.update(make_table())
        finally:
            stop_refresh.set()
            refresh_thread.join(timeout=1)
    
    # Combine and save
    console.print(f"\n[bold cyan]Saving combined transcription...[/bold cyan]")
    
    combined_text = "\n\n".join(all_transcriptions).strip()
    output_file.write_text(combined_text, encoding='utf-8')
    
    console.print(f"[bold green]✓[/bold green] Saved to: {output_file}")
    
    # Final stats
    # Calculate total processing time and speed
    batch_elapsed_time = time.time() - batch_start_time
    speed_ratio = total_duration / batch_elapsed_time if batch_elapsed_time > 0 else 0
    
    console.print("\n[bold cyan]Summary:[/bold cyan]")
    summary_table = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Files processed", str(len(audio_files)))
    summary_table.add_row("Total duration", format_duration(total_duration))
    summary_table.add_row("Processing time", format_duration(batch_elapsed_time))
    summary_table.add_row("Speed", f"{speed_ratio:.1f}x realtime")
    summary_table.add_row("Total words", f"{total_words:,}")
    summary_table.add_row("Total characters", f"{total_chars:,}")
    console.print(summary_table)
    
    console.print("\n[bold green]Batch transcription completed successfully![/bold green]\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    %(prog)s input.mp3 output.txt
    %(prog)s audio.wav transcript.txt --language en
    %(prog)s lecture.mp3 transcript.txt --language de --model large

  Batch mode (directory input):
    %(prog)s ./recordings/ combined_output.txt --language de
    %(prog)s /path/to/lectures/ transcript.txt --model medium

Available models:
  turbo  - Large-v3-turbo, ~800MB, 8x faster, multilingual (default)
  large  - Best accuracy, ~3GB
  medium - Good balance, ~1.5GB, 2-3x faster than large
  small  - Fast, ~466MB, lower accuracy
  base   - Very fast, ~145MB, basic accuracy

Batch mode:
  When input is a directory, all audio files are processed in natural
  sort order and combined into a single output file.
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file or directory (for batch mode)"
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Output text file for transcription"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["large", "turbo", "medium", "small", "base"],
        default="turbo",
        help="Whisper model size (default: turbo)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'ru', 'de', 'es'). If not specified, will auto-detect."
    )

    args = parser.parse_args()

    try:
        # Check if input is directory (batch mode) or single file
        if args.input.is_dir():
            transcribe_batch(args.input, args.output_file, args.language, args.model)
        else:
            transcribe_audio(args.input, args.output_file, args.language, args.model)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]! Operation cancelled by user[/yellow]")
        console.print("[dim]Exiting gracefully...[/dim]\n")
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
