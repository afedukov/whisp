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
    "large-v2": "large-v2",
    "turbo": "large-v3-turbo",  # 8x faster than large-v3, multilingual (de, en, ru, etc.)
    "medium": "medium",
    "small": "small",
    "base": "base"
}

MODEL_INFO = {
    "large": "Best accuracy, ~3GB",
    "large-v2": "Previous large version, ~3GB, slightly faster",
    "turbo": "Large-v3-turbo, ~800MB, 8x faster, multilingual",
    "medium": "Good balance, ~1.5GB",
    "small": "Fast, ~466MB, lower accuracy",
    "base": "Very fast, ~145MB, basic accuracy"
}

# Default translation prompt for OpenAI API
# Default system prompt for translation
DEFAULT_SYSTEM_PROMPT = "You are a professional translator and editor specializing in academic lectures and public speaking."

# Default user prompt template
DEFAULT_USER_PROMPT = """Task: Translate the following transcript of a spoken lecture from {source_language} to {target_language}.

Context: The source text is an automated transcription (ASR) of a speech. It contains phonetic errors, 
misheard words, and run-on sentences typical of spoken language.

Instructions:
1.  **Correct & Translate:** Translate the text into natural, fluent {target_language}. If you encounter 
    obvious transcription errors (words that sound similar but make no sense in context, e.g., "Militärgisse" 
    or "brille vor Angst"), reconstruct the intended meaning based on the context before translating.
2.  **Style & Tone:** Maintain the speaker's rhetorical style (storytelling, engaging, slightly informal 
    but educational). Avoid word-for-word translation. Rephrase sentences to sound natural in {target_language}, 
    as if a native speaker were giving the lecture.
3.  **Structure:** Organize the text into logical paragraphs to improve readability. Fix punctuation where 
    the transcript is messy.
4.  **Accuracy:** Preserve all names (George Bush, Al Gore, Pat Buchanan), dates, and numbers exactly.
5.  **Output:** Provide ONLY the translated text, nothing else.

Text to translate:
{text}"""

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
    },
    "recording": {
        "sample_rate": 16000,
        "channels": 1,
        "default_device": -1,
        "show_level_meter": True,
        "save_dir": "",
        "keep_recording": False,
        "compress_format": "m4a"
    },
    "translation": {
        "openai_api_key": "",
        "model": "gpt-5-mini",
        "temperature": 0.3,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT
    }
}

def load_config() -> dict:
    """Load configuration from config.yaml in the app directory.
    Falls back to default values if file not found or invalid.
    """
    import yaml
    
    config = DEFAULT_CONFIG.copy()
    
    # 1. App directory config (default/global)
    app_config_path = Path(__file__).parent / "config.yaml"
    
    # 2. User home config (~/.whisp/config.yaml)
    home_config_path = Path.home() / ".whisp" / "config.yaml"
    
    # Load in order of priority (lowest to highest)
    config_paths = [
        (app_config_path, "App"),
        (home_config_path, "Home")
    ]
    
    for path, source in config_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                
                if user_config:
                    # Deep merge user config with defaults/previous
                    for section in DEFAULT_CONFIG:
                        if section in user_config and isinstance(user_config[section], dict):
                            config[section] = {**config.get(section, DEFAULT_CONFIG[section]), **user_config[section]}
                    
                    # console might not be fully initialized or we want to be subtle, 
                    # but since console is global, we can try using it if we really want debug info
                    # For now, silent success is best, unless debugging.
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load {source} config ({path}): {e}[/yellow]")

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


def compress_audio_to_m4a(wav_file: Path, output_path: Path = None, quiet: bool = False) -> Path:
    """Compress WAV audio to M4A format using ffmpeg.

    M4A with AAC codec provides ~10x size reduction compared to WAV
    with minimal quality loss for speech.

    Args:
        wav_file: Path to input WAV file
        output_path: Optional output path for M4A file. If None, creates temp file.
        quiet: If True, suppress output messages

    Returns:
        Path to compressed M4A file
    """
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        console.print("[bold red]✗[/bold red] ffmpeg not found in PATH")
        console.print("[yellow]Please install ffmpeg to compress audio files[/yellow]")
        console.print("[dim]macOS: brew install ffmpeg[/dim]")
        sys.exit(1)

    if not quiet:
        console.print(f"\n[cyan]Compressing to M4A format...[/cyan]")

    try:
        # Use provided output path or create temporary M4A file
        if output_path:
            temp_m4a_path = output_path
        else:
            temp_m4a = tempfile.NamedTemporaryFile(suffix='.m4a', delete=False)
            temp_m4a_path = Path(temp_m4a.name)
            temp_m4a.close()

        # Compress using ffmpeg with AAC codec
        # -i: input file
        # -c:a aac: use AAC audio codec
        # -b:a 64k: bitrate 64kbps (good for speech, very small files)
        # -y: overwrite output file
        result = subprocess.run(
            [
                'ffmpeg',
                '-i', str(wav_file),
                '-c:a', 'aac',
                '-b:a', '64k',
                '-y',
                str(temp_m4a_path)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            console.print(f"[bold red]✗[/bold red] FFmpeg compression failed")
            console.print(f"[dim]{result.stderr}[/dim]")
            raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

        if not quiet:
            # Show size comparison
            wav_size = wav_file.stat().st_size
            m4a_size = temp_m4a_path.stat().st_size
            ratio = (wav_size / m4a_size) if m4a_size > 0 else 1
            console.print(f"[green]✓[/green] Compressed: {wav_size / (1024*1024):.1f}MB → {m4a_size / (1024*1024):.1f}MB ({ratio:.1f}x smaller)")

        return temp_m4a_path

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Audio compression failed: {str(e)}")
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


def transcribe_audio(audio_file: Path, output_file: Path, language: str = None, model_size: str = "large", translate_to: str = None, show_save_message: bool = True):
    """Transcribe audio file to text (displayed as batch mode with 1 file)"""
    from rich.table import Table
    from rich.live import Live

    print_header(model_size)

    # Check if input file exists
    if not check_file_exists(audio_file):
        sys.exit(1)

    console.print(f"\n[dim]Input: {audio_file.resolve()}[/dim]")
    console.print(f"[dim]Output: {output_file.resolve()}[/dim]")

    # Determine device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[dim]Device: {device}[/dim]")
    except ImportError:
        device = "cpu"
        console.print(f"[dim]Device: {device}[/dim]")

    # Load model
    try:
        model = load_model(device, model_size)
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Failed to load model: {str(e)}")
        sys.exit(1)

    # Get audio duration
    audio_duration = get_audio_duration(audio_file)

    # Initialize status tracking
    file_status = [""]  # "" = pending, "spinner MM:SS" = processing, "✓ N words" = done
    processing_start_time = [0.0]

    import time
    import threading

    # Spinner animation frames
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_index = [0]

    def make_table():
        """Create progress table for single file"""
        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Duration", style="green", justify="right", width=8)
        table.add_column("Status", justify="left", width=16)

        dur_str = format_duration(audio_duration) if audio_duration > 0 else "?"

        # Show spinner if processing
        if processing_start_time[0] > 0 and file_status[0] == "":
            elapsed = time.time() - processing_start_time[0]
            elapsed_str = format_duration(elapsed)
            spinner = spinner_frames[spinner_index[0] % len(spinner_frames)]
            status = f"[yellow]{spinner} {elapsed_str}[/yellow]"
        else:
            status = file_status[0]

        table.add_row("1", audio_file.name, dur_str, status)
        return table

    stop_refresh = threading.Event()

    def refresh_loop(live):
        """Background thread to animate spinner"""
        while not stop_refresh.is_set():
            spinner_index[0] += 1
            live.update(make_table())
            time.sleep(0.1)

    # Process with live table
    console.print(f"\n[bold cyan]Processing...[/bold cyan]")

    batch_start_time = time.time()

    with Live(make_table(), console=console, refresh_per_second=4) as live:
        # Start background refresh thread
        refresh_thread = threading.Thread(target=refresh_loop, args=(live,), daemon=True)
        refresh_thread.start()

        try:
            # Start processing
            processing_start_time[0] = time.time()
            live.update(make_table())

            # Transcribe
            transcription, info = transcribe_single_file(model, audio_file, language, quiet=True)

            # Update status to done
            word_count = len(transcription.split())
            file_status[0] = f"[green]✓ {word_count} words[/green]"
            processing_start_time[0] = 0
            live.update(make_table())

        except Exception as e:
            file_status[0] = f"[red]✗ error[/red]"
            processing_start_time[0] = 0
            live.update(make_table())
            raise
        finally:
            stop_refresh.set()
            refresh_thread.join(timeout=1)

    # Save transcription
    output_file.write_text(transcription, encoding='utf-8')

    # Display preview of transcription
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

    # Translation (if requested)
    translation = None
    translation_file = None
    translation_time = 0

    if translate_to:
        try:
            translation_start = time.time()
            translation = translate_with_openai(
                text=transcription,
                target_language=translate_to,
                source_language=info.get('language', 'auto')
            )
            translation_time = time.time() - translation_start

            # Save translation to basename_<lang>.txt
            translation_file = get_translation_filename(output_file, translate_to)
            translation_file.write_text(translation, encoding='utf-8')

            # Display preview of translation
            translation_preview = translation[:preview_length]
            if len(translation) > preview_length:
                translation_preview += "..."

            console.print(f"\n[bold cyan]Translation Preview:[/bold cyan]")
            console.print(Panel(
                translation_preview,
                box=box.ROUNDED,
                border_style="green",
                padding=(1, 2)
            ))

        except Exception as e:
            console.print(f"\n[red]✗ Translation failed:[/red] {e}")
            console.print(f"[yellow]Transcription saved without translation[/yellow]")
            translation = None
            translation_file = None

    # Stats
    processing_time = time.time() - batch_start_time
    speed_ratio = audio_duration / processing_time if processing_time > 0 and audio_duration > 0 else 0
    char_count = len(transcription)

    console.print("\n[bold cyan]Stats:[/bold cyan]")
    stats_table = Table(box=box.ROUNDED, border_style="green", show_header=False)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="bold")
    stats_table.add_row("Duration", format_duration(audio_duration) if audio_duration > 0 else "unknown")
    stats_table.add_row("Processing time", format_duration(processing_time))
    stats_table.add_row("Speed", f"{speed_ratio:.1f}x realtime")
    stats_table.add_row("Words", f"{word_count:,}")
    stats_table.add_row("Characters", f"{char_count:,}")
    if translation_time > 0:
        stats_table.add_row("Translation time", format_duration(translation_time))
    console.print(stats_table)

    if show_save_message:
        console.print("\n[bold green]Transcription completed successfully[/bold green]")
        console.print(f"[dim]└──[/dim] Transcription saved to: {output_file.resolve()}")
        if translation_file:
            console.print(f"[dim]└──[/dim] Translation saved to: {translation_file.resolve()}")


# =============================================================================
# Recording Mode Functions
# =============================================================================

def list_audio_devices() -> list[dict]:
    """List available audio input devices.

    Returns:
        List of dicts with keys: index, name, channels, sample_rate

    Raises:
        SystemExit: If no devices found or permission denied
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception as e:
        # macOS permission error or sounddevice not installed
        error_str = str(e).lower()
        if "permission" in error_str or "access" in error_str:
            console.print("[bold red]✗[/bold red] Cannot access audio devices")
            console.print("\n[yellow]macOS: Grant microphone permission[/yellow]")
            console.print("[dim]System Settings → Privacy & Security → Microphone → Terminal[/dim]")
            console.print("[dim]Then restart this script[/dim]\n")
        else:
            console.print(f"[bold red]✗[/bold red] Audio device error: {e}")
            console.print("[yellow]Make sure sounddevice is installed: pip install sounddevice[/yellow]")
        sys.exit(1)

    # Filter to input devices only
    input_devices = []
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append({
                'index': idx,
                'name': d['name'],
                'channels': d['max_input_channels'],
                'sample_rate': d['default_samplerate']
            })

    if not input_devices:
        console.print("[bold red]✗[/bold red] No microphones found")
        console.print("[yellow]Please connect a microphone and try again[/yellow]")
        sys.exit(1)

    return input_devices


def select_audio_device(devices: list[dict]) -> int:
    """Interactive device selection with arrow keys and beautiful styling.

    Args:
        devices: List of device dicts from list_audio_devices()

    Returns:
        Device index to use, or -1 for system default
    """
    from simple_term_menu import TerminalMenu

    # Print header
    console.print("\n[bold cyan]Available Microphones[/bold cyan]")
    console.print("[dim]Use arrow keys ↑↓ to navigate, Enter to select[/dim]\n")

    # Format table-like entries with fixed column widths
    # Column widths: Device Name (50), Channels (10), Index (7)
    device_col_width = 50
    channels_col_width = 10
    index_col_width = 7

    # Create formatted entries
    menu_entries = []
    index_map = []

    # Header row (non-selectable, will be shown as title)
    header = (
        f"{'Device Name':<{device_col_width}} "
        f"{'Channels':^{channels_col_width}} "
        f"{'Index':^{index_col_width}}"
    )
    separator = "─" * (device_col_width + channels_col_width + index_col_width + 2)

    # Add System Default
    default_entry = (
        f"{'System Default':<{device_col_width}} "
        f"{'-':^{channels_col_width}} "
        f"{'-':^{index_col_width}}"
    )
    menu_entries.append(default_entry)
    index_map.append(-1)

    # Add all devices
    for d in devices:
        name = d['name']
        if len(name) > device_col_width:
            name = name[:device_col_width - 3] + "..."

        entry = (
            f"{name:<{device_col_width}} "
            f"{str(d['channels']):^{channels_col_width}} "
            f"{str(d['index']):^{index_col_width}}"
        )
        menu_entries.append(entry)
        index_map.append(d['index'])

    # Create terminal menu with cyan theme and header
    terminal_menu = TerminalMenu(
        menu_entries,
        title=f"{header}\n{separator}",
        menu_cursor="→ ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("fg_cyan", "bold"),
        cycle_cursor=True,
        clear_screen=False
    )

    try:
        # Show menu and get selection
        menu_entry_index = terminal_menu.show()

        console.print()  # Empty line for spacing

        if menu_entry_index is not None:
            return index_map[menu_entry_index]
        else:
            # User cancelled (Ctrl+C or ESC)
            return -1
    except KeyboardInterrupt:
        console.print("\n")
        return -1


def generate_timestamped_filename(suffix: str, prefix: str = "recording") -> str:
    """Generate filename with timestamp.

    Args:
        suffix: File extension (e.g., '.txt', '.m4a')
        prefix: Filename prefix (default: 'recording')

    Returns:
        Filename string like 'recording_20251208_185239.txt'
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}"


def get_recording_file_path(suffix: str = '.wav', base_filename: str = None) -> Path:
    """Generate path for recording file based on config.

    If save_dir is configured and keep_recording is True, creates a timestamped
    file in save_dir. Otherwise, creates a temporary file.

    Args:
        suffix: File extension (e.g., '.wav', '.m4a')
        base_filename: Optional base filename (e.g., 'recording_20251208_185239')
                      If None, generates new timestamp

    Returns:
        Path object for the recording file
    """
    config = CONFIG["recording"]
    save_dir = config.get("save_dir", "")
    keep_recording = config.get("keep_recording", False)

    if save_dir and keep_recording:
        # Create permanent file with timestamp
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if base_filename:
            filename = f"{base_filename}{suffix}"
        else:
            filename = generate_timestamped_filename(suffix)
        return save_path / filename
    else:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        return temp_path


def record_audio_fixed_duration(device_index: int, duration: float, base_filename: str = None) -> Path:
    """Record audio for a fixed duration with progress bar.

    Args:
        device_index: Device index from select_audio_device(), or -1 for default
        duration: Recording duration in seconds
        base_filename: Optional base filename (e.g., 'recording_20251208_185239')

    Returns:
        Path to temporary audio file (WAV or M4A depending on config)

    Raises:
        SystemExit: If recording fails
    """
    import sounddevice as sd
    import soundfile as sf

    config = CONFIG["recording"]
    sample_rate = config["sample_rate"]
    channels = config["channels"]

    # Create WAV file (temp or permanent based on config)
    temp_path = get_recording_file_path('.wav', base_filename=base_filename)

    console.print(f"\n[cyan]Recording for {duration:.0f} seconds...[/cyan]")

    try:
        from rich.progress import BarColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Recording...", total=duration)

            # Record with sounddevice
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                device=device_index if device_index >= 0 else None,
                dtype='int16'
            )

            # Update progress while recording
            import time
            elapsed = 0.0
            while elapsed < duration:
                time.sleep(0.1)
                elapsed += 0.1
                progress.update(task, completed=min(elapsed, duration))

            sd.wait()  # Wait for recording to complete
            progress.update(task, completed=duration)

        # Save to WAV file
        sf.write(temp_path, recording, sample_rate)
        console.print("[green]✓[/green] Recording complete")

        # Compress to M4A if configured
        compress_format = config.get("compress_format", "wav").lower()
        if compress_format == "m4a":
            try:
                # Generate M4A path (replace .wav with .m4a)
                m4a_path = temp_path.with_suffix('.m4a')
                compress_audio_to_m4a(temp_path, output_path=m4a_path, quiet=False)
                # Delete the WAV file after successful compression
                temp_path.unlink()
                return m4a_path
            except Exception as e:
                console.print(f"[yellow]Warning: Compression failed, using WAV: {e}[/yellow]")
                return temp_path
        else:
            return temp_path

    except KeyboardInterrupt:
        console.print("\n[yellow]! Recording cancelled[/yellow]")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Recording failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)


def record_audio_interactive(device_index: int, base_filename: str = None) -> Path:
    """Record audio with manual start/stop control (Ctrl+D).

    Args:
        device_index: Device index from select_audio_device(), or -1 for default
        base_filename: Optional base filename (e.g., 'recording_20251208_185239')

    Returns:
        Path to temporary audio file (WAV or M4A depending on config)

    Raises:
        SystemExit: If recording fails or is cancelled
    """
    import sounddevice as sd
    import soundfile as sf
    import threading
    import numpy as np

    config = CONFIG["recording"]
    sample_rate = config["sample_rate"]
    channels = config["channels"]

    # Create WAV file (temp or permanent based on config)
    temp_path = get_recording_file_path('.wav', base_filename=base_filename)

    console.print("\n[dim]Press ENTER to start recording...[/dim]")
    input()

    # Recording state
    recording_data = []
    stop_event = threading.Event()
    current_level = [0.0]  # Current audio level (0.0 to 1.0)

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio block"""
        if status:
            console.print(f"[yellow]Status: {status}[/yellow]")
        recording_data.append(indata.copy())

        # Calculate RMS level for visualization
        if config.get("show_level_meter", True):
            # Convert int16 to float and calculate RMS
            audio_float = indata.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float ** 2))
            current_level[0] = min(rms * 3.0, 1.0)  # Scale and clamp to [0, 1]

    # No separate input thread needed - we'll check in the main loop

    try:
        from rich.live import Live
        from rich.panel import Panel
        import time
        import sys
        import select
        import tty
        import termios

        # Save original terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to cbreak mode (no echo, but process signals normally)
            tty.setcbreak(sys.stdin.fileno())

            start_time = time.time()

            # Pulsing animation states
            pulse_states = [
                "[bold red]●[/bold red]",
                "[red]●[/red]",
                "[bold red]●[/bold red]",
                "[red]○[/red]"
            ]

            with sd.InputStream(
                device=device_index if device_index >= 0 else None,
                channels=channels,
                samplerate=sample_rate,
                dtype='int16',
                callback=audio_callback
            ):
                # Create live panel with pulsing red dot
                pulse_idx = 0

                with Live(console=console, refresh_per_second=4, transient=False) as live:
                    while not stop_event.is_set():
                        # Check for Ctrl+D (non-blocking)
                        ready, _, _ = select.select([sys.stdin], [], [], 0)
                        if ready:
                            try:
                                char = sys.stdin.read(1)
                                if char == '\x04':  # Ctrl+D (EOT)
                                    stop_event.set()
                                    break
                            except:
                                pass

                        elapsed = time.time() - start_time
                        duration_str = format_duration(elapsed)

                        # Calculate file size in real-time
                        # sample_rate * channels * bytes_per_sample (int16 = 2 bytes) * duration
                        file_size_bytes = sample_rate * channels * 2 * elapsed
                        if file_size_bytes < 1024 * 1024:  # Less than 1 MB
                            file_size_str = f"{file_size_bytes / 1024:.1f} KB"
                        else:
                            file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"

                        # Pulsing red dot
                        dot = pulse_states[pulse_idx % len(pulse_states)]
                        pulse_idx += 1

                        # Build panel content
                        panel_content = f"{dot} [bold]RECORDING[/bold]\n"
                        panel_content += f"[dim]Duration: {duration_str}[/dim]\n"
                        panel_content += f"[dim]File size: {file_size_str}[/dim]\n"

                        # Add level meter if enabled
                        if config.get("show_level_meter", True):
                            level = current_level[0]

                            # Progress bar ████░░░░
                            bar_length = 30
                            filled = int(level * bar_length)
                            if level > 0.9:
                                bar_color = "red"
                            elif level > 0.7:
                                bar_color = "yellow"
                            else:
                                bar_color = "green"

                            progress_bar = f"[{bar_color}]{'█' * filled}[/{bar_color}]"
                            progress_bar += f"[dim]{'░' * (bar_length - filled)}[/dim]"

                            panel_content += f"\n[dim]Level:[/dim] {progress_bar}\n"

                        panel_content += f"\n[dim]Press Ctrl+D to stop[/dim]"

                        # Update live display
                        live.update(
                            Panel(
                                panel_content,
                                border_style="red",
                                box=box.ROUNDED
                            )
                        )
                        time.sleep(0.25)
        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        console.print("\n[green]✓[/green] Recording stopped")

        # Combine all chunks and save
        if not recording_data:
            console.print("[yellow]Warning: No audio data recorded[/yellow]")
            if temp_path.exists():
                temp_path.unlink()
            sys.exit(1)

        full_recording = np.concatenate(recording_data, axis=0)
        sf.write(temp_path, full_recording, sample_rate)

        # Compress to M4A if configured
        compress_format = config.get("compress_format", "wav").lower()
        if compress_format == "m4a":
            try:
                # Generate M4A path (replace .wav with .m4a)
                m4a_path = temp_path.with_suffix('.m4a')
                compress_audio_to_m4a(temp_path, output_path=m4a_path, quiet=False)
                # Delete the WAV file after successful compression
                temp_path.unlink()
                return m4a_path
            except Exception as e:
                console.print(f"[yellow]Warning: Compression failed, using WAV: {e}[/yellow]")
                return temp_path
        else:
            return temp_path

    except KeyboardInterrupt:
        console.print("\n[yellow]! Recording cancelled[/yellow]")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Recording failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)


def record_and_transcribe(
    output_file: Path = None,
    language: str = None,
    model_size: str = "turbo",
    translate_to: str = None
):
    """Main recording mode orchestrator.

    Manages the full recording workflow:
    1. List and select microphone
    2. Record audio (manual or fixed duration)
    3. Transcribe using existing transcribe_audio()
    4. Clean up temporary files

    Args:
        output_file: Path to save transcription (optional, auto-generated if None)
        language: Language code (e.g., 'en', 'de') or None for auto-detect
        model_size: Whisper model to use
        translate_to: Target language code for translation (e.g., 'ru', 'en') or None to skip translation
    """
    import shutil

    # Generate timestamp for file naming (used for both audio and txt if output_file=None)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If output_file not specified, auto-generate in save_dir
    if output_file is None:
        save_dir = CONFIG["recording"].get("save_dir", "")
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f"recording_{timestamp}.txt"
        else:
            # No save_dir configured, use current directory
            output_file = Path(f"recording_{timestamp}.txt")

    # Header
    print_header(model_size)
    console.print("\n[bold cyan]RECORDING MODE[/bold cyan]\n")

    # Check disk space
    temp_dir = Path(tempfile.gettempdir())
    try:
        stat = shutil.disk_usage(temp_dir)
        if stat.free < 100 * 1024 * 1024:  # Less than 100MB
            console.print("[bold red]✗[/bold red] Insufficient disk space")
            console.print(f"[yellow]Free space: {stat.free / (1024*1024):.1f} MB[/yellow]")
            console.print("[dim]Need at least 100 MB for recording[/dim]")
            sys.exit(1)
    except:
        pass  # Skip disk check if it fails

    # Step 1: List and select device (or use from config)
    config_device = CONFIG["recording"].get("default_device", -1)
    devices = list_audio_devices()
    device_index = -1

    # Check if device is specified in config
    if config_device != -1 and config_device is not None:
        if isinstance(config_device, str):
            # Device specified by name - search for it
            search_name = config_device.lower()
            matching_device = None

            # Try exact match first
            for d in devices:
                if d['name'].lower() == search_name:
                    matching_device = d
                    break

            # If no exact match, try partial match
            if not matching_device:
                for d in devices:
                    if search_name in d['name'].lower():
                        matching_device = d
                        break

            if matching_device:
                device_index = matching_device['index']
                device_name = matching_device['name']
                if len(device_name) > 50:
                    device_name = device_name[:47] + "..."
                console.print(f"\n[dim]Using microphone: [green]{device_name}[/green] (from config)[/dim]")
            else:
                console.print(f"\n[yellow]Warning: Device '{config_device}' from config not found[/yellow]")
                console.print("[dim]Available devices:[/dim]")
                for d in devices:
                    console.print(f"[dim]  - {d['name']}[/dim]")
                console.print("[dim]Falling back to device selection...[/dim]")
                device_index = select_audio_device(devices)

        elif isinstance(config_device, int):
            # Device specified by index (legacy support)
            device_index = config_device
            device_name = next((d['name'] for d in devices if d['index'] == device_index), None)

            if device_name:
                if len(device_name) > 50:
                    device_name = device_name[:47] + "..."
                console.print(f"\n[dim]Using microphone: [green]{device_name}[/green] (from config)[/dim]")
            else:
                console.print(f"\n[yellow]Warning: Device index {device_index} from config not found[/yellow]")
                console.print("[dim]Falling back to device selection...[/dim]")
                device_index = select_audio_device(devices)
    else:
        # No device specified in config - interactive selection
        device_index = select_audio_device(devices)

        if device_index == -1:
            console.print("[dim]Using system default microphone[/dim]")
        else:
            device_name = next((d['name'] for d in devices if d['index'] == device_index), "Unknown")
            if len(device_name) > 50:
                device_name = device_name[:47] + "..."
            console.print(f"[dim]Selected: [green]{device_name}[/green][/dim]")

    # Step 2: Record audio (manual mode only)
    temp_wav_path = None

    # Generate base filename for consistent naming between audio and txt
    base_filename = f"recording_{timestamp}"

    success = False
    try:
        # Record with manual start/stop
        temp_wav_path = record_audio_interactive(device_index, base_filename=base_filename)

        # Step 3: Transcribe (reuse existing function with silent save)
        console.print("\n[cyan]Recording complete! Starting transcription...[/cyan]")
        transcribe_audio(temp_wav_path, output_file, language, model_size, translate_to, show_save_message=False)
        success = True

    except Exception:
        # On failure, still notify if recording is kept
        if temp_wav_path and temp_wav_path.exists():
            if CONFIG["recording"].get("keep_recording", False):
                console.print(f"\n[bold cyan]Saving recording...[/bold cyan]")
                console.print(f"[bold green]✓[/bold green] Saved to: {temp_wav_path.resolve()}")
        raise

    finally:
        # Step 4: Cleanup (only delete if NOT keeping)
        if temp_wav_path and temp_wav_path.exists():
            if not CONFIG["recording"].get("keep_recording", False):
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass  # Ignore cleanup errors

    # Print consolidated summary on success
    if success:
        console.print("\n[bold green]Transcription completed successfully[/bold green]")
        
        if CONFIG["recording"].get("keep_recording", False):
            console.print(f"[dim]├──[/dim] Recording saved to: {temp_wav_path.resolve()}")
            console.print(f"[dim]└──[/dim] Transcription saved to: {output_file.resolve()}")
        else:
            console.print(f"[dim]└──[/dim] Transcription saved to: {output_file.resolve()}")


# =============================================================================
# Translation Functions (OpenAI API)
# =============================================================================

def translate_with_openai(text: str, target_language: str, source_language: str = "auto") -> str:
    """
    Translate text using OpenAI ChatGPT API.

    Args:
        text: Text to translate
        target_language: Target language code (e.g., 'ru', 'en', 'de')
        source_language: Source language (detected from transcription)

    Returns:
        Translated text with paragraphs

    Raises:
        Exception if translation fails
    """
    from openai import OpenAI
    import os
    import time

    config = CONFIG.get("translation", {})
    api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model = config.get("model", "gpt-5-mini")
    temperature = config.get("temperature", 0.3)
    
    system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_prompt_template = config.get("user_prompt", DEFAULT_USER_PROMPT)

    if not api_key:
        raise Exception(
            "OpenAI API key not configured. "
            "Add 'openai_api_key' to translation section in config.yaml "
            "or set OPENAI_API_KEY environment variable"
        )

    # Language code to full name mapping
    LANG_NAMES = {
        'en': 'English',
        'ru': 'Russian',
        'de': 'German',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'tr': 'Turkish',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }

    target_lang_name = LANG_NAMES.get(target_language.lower(), target_language)
    source_lang_name = LANG_NAMES.get(source_language.lower(), source_language)

    # Format user prompt with target language
    formatted_user_prompt = user_prompt_template.format(
        source_language=source_lang_name,
        target_language=target_lang_name,
        text=text
    )

    try:
        from rich.live import Live
        import threading

        client = OpenAI(api_key=api_key)

        # Spinner animation for translation
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_index = [0]
        stop_spinner = threading.Event()
        translation_start_time = time.time()

        def spinner_message():
            elapsed = time.time() - translation_start_time
            elapsed_str = format_duration(elapsed)
            spinner = spinner_frames[spinner_index[0] % len(spinner_frames)]
            return f"\n[yellow]{spinner} Translating to {target_lang_name} using [green]{model}[/green]... {elapsed_str}[/yellow]"

        def refresh_spinner():
            """Background thread to animate spinner"""
            while not stop_spinner.is_set():
                spinner_index[0] += 1
                time.sleep(0.1)

        # Start spinner animation in background
        spinner_thread = threading.Thread(target=refresh_spinner, daemon=True)
        spinner_thread.start()

        with Live(spinner_message(), console=console, refresh_per_second=10) as live:
            def update_spinner():
                while not stop_spinner.is_set():
                    live.update(spinner_message())
                    time.sleep(0.1)

            update_thread = threading.Thread(target=update_spinner, daemon=True)
            update_thread.start()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ],
                temperature=temperature
            )

            translation = response.choices[0].message.content.strip()

            # Stop spinner
            stop_spinner.set()
            spinner_thread.join(timeout=0.5)

        console.print(f"[green]✓[/green] Translation complete")

        return translation

    except Exception as e:
        error_msg = str(e).lower()

        if "api key" in error_msg or "unauthorized" in error_msg:
            raise Exception("OpenAI API key is invalid")
        elif "quota" in error_msg or "insufficient" in error_msg:
            raise Exception(
                "OpenAI API quota exceeded. "
                "Add credits at https://platform.openai.com/account/billing"
            )
        elif "model" in error_msg:
            # Include original error as it might be an access issue, not just a typo
            raise Exception(f"Model '{model}' error: {e}. Check config.yaml or your OpenAI account access.")
        else:
            raise Exception(f"OpenAI API error: {e}")


def get_translation_filename(original_path: Path, target_language: str) -> Path:
    """
    Generate filename for translation.

    Examples:
        lecture.txt + 'ru' → lecture_ru.txt
        recording_20251209_072149.txt + 'en' → recording_20251209_072149_en.txt

    Args:
        original_path: Path to original transcription file
        target_language: Language code (e.g., 'ru', 'en')

    Returns:
        Path for translation file
    """
    stem = original_path.stem
    suffix = original_path.suffix
    parent = original_path.parent

    # Create new filename: basename_<lang>.txt
    new_stem = f"{stem}_{target_language.lower()}"

    return parent / f"{new_stem}{suffix}"


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


def transcribe_batch(input_dir: Path, output_file: Path, language: str = None, model_size: str = "large", translate_to: str = None):
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
    console.print(f"[dim]Directory: {input_dir.resolve()}[/dim]")
    console.print(f"[dim]Output: {output_file.resolve()}[/dim]")
    
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
    # Status: "" = pending, "spinner MM:SS" = in progress with elapsed time, "✓ N words" = done, "✗ error" = failed
    file_statuses = ["" for _ in audio_files]
    current_file_index = [-1]  # Use list to allow mutation from inner function
    processing_start_time = [0.0]

    import time
    import threading

    # Spinner animation frames
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_index = [0]  # Use list to allow mutation

    def make_table():
        """Create the progress table with current statuses"""
        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Duration", style="green", justify="right", width=8)
        table.add_column("Status", justify="left", width=16)

        for i, (f, dur, status) in enumerate(zip(audio_files, file_durations, file_statuses), 1):
            dur_str = format_duration(dur) if dur > 0 else "?"
            # If this is the currently processing file, show animated spinner with elapsed time
            if i - 1 == current_file_index[0] and processing_start_time[0] > 0:
                elapsed = time.time() - processing_start_time[0]
                elapsed_str = format_duration(elapsed)
                # Get current spinner frame
                spinner = spinner_frames[spinner_index[0] % len(spinner_frames)]
                status = f"[yellow]{spinner} {elapsed_str}[/yellow]"
            table.add_row(str(i), f.name, dur_str, status)

        return table
    
    # Process files with live table
    console.print(f"\n[bold cyan]Processing...[/bold cyan]")
    
    all_transcriptions = []
    all_translations = []  # For storing translations if requested
    total_words = 0
    total_chars = 0
    batch_start_time = time.time()  # Track total processing time
    
    stop_refresh = threading.Event()
    
    def refresh_loop(live):
        """Background thread to refresh elapsed time and animate spinner"""
        while not stop_refresh.is_set():
            spinner_index[0] += 1  # Advance spinner animation
            live.update(make_table())
            time.sleep(0.1)  # Faster refresh for smooth spinner animation
    
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

                    # Translate if requested
                    if translate_to:
                        try:
                            file_statuses[i] = f"[yellow]↻ translating...[/yellow]"
                            live.update(make_table())

                            translation = translate_with_openai(
                                text=transcription,
                                target_language=translate_to,
                                source_language=info.get('language', 'auto')
                            )
                            all_translations.append(translation)
                        except Exception as e:
                            console.print(f"\n[red]✗ Translation failed for {audio_file.name}:[/red] {e}")
                            all_translations.append(f"[TRANSLATION ERROR: {str(e)}]")

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
    
    console.print(f"[bold green]✓[/bold green] Saved to: {output_file.resolve()}")

    # Save combined translation if requested
    if translate_to and all_translations:
        translation_file = get_translation_filename(output_file, translate_to)
        combined_translation = "\n\n".join(all_translations).strip()
        translation_file.write_text(combined_translation, encoding='utf-8')
        console.print(f"[bold green]✓[/bold green] [bold cyan]Translation saved:[/bold cyan] [green]{translation_file.resolve()}[/green]")

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
    %(prog)s input.mp3                    # Auto-generates input.txt
    %(prog)s audio.wav --language en      # Auto-generates audio.txt
    %(prog)s lecture.mp3 transcript.txt --language de --model large

  Batch mode (directory input):
    %(prog)s ./recordings/ combined_output.txt --language de
    %(prog)s ./recordings/                # Auto-generates recordings.txt
    %(prog)s /path/to/lectures/ --model medium

  Recording mode:
    %(prog)s record                       # Auto-saves to save_dir/recording_TIMESTAMP.txt
    %(prog)s record output.txt
    %(prog)s record transcript.txt --model turbo --language de

Available models:
  turbo    - Large-v3-turbo, ~800MB, 8x faster, multilingual (default)
  large    - Best accuracy (v3), ~3GB
  large-v2 - Previous large version, ~3GB, slightly faster
  medium   - Good balance, ~1.5GB, 2-3x faster than large
  small    - Fast, ~466MB, lower accuracy
  base     - Very fast, ~145MB, basic accuracy

Modes:
  - Single file: Transcribe one audio file
  - Batch: When input is a directory, all audio files are processed in natural
    sort order and combined into a single output file
  - Recording: Use "record" as input to record from microphone and transcribe
        """
    )

    parser.add_argument(
        "input",
        type=str,  # Changed from Path to str to allow "record" keyword
        help="Input audio file, directory (batch mode), or 'record' (recording mode)"
    )

    parser.add_argument(
        "output_file",
        type=Path,
        nargs='?',  # Make output_file optional
        default=None,
        help="Output text file for transcription (optional, auto-generated if not specified)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["large", "large-v2", "turbo", "medium", "small", "base"],
        default="turbo",
        help="Whisper model size (default: turbo)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'ru', 'de', 'es'). If not specified, will auto-detect."
    )

    parser.add_argument(
        "--translate",
        type=str,
        default=None,
        metavar="LANG",
        help="Translate transcription to target language using OpenAI API (e.g., 'ru', 'en', 'de')"
    )

    args = parser.parse_args()

    try:
        # Check mode: recording, batch (directory), or single file
        if args.input.lower() == "record":
            # Recording mode
            record_and_transcribe(args.output_file, args.language, args.model, args.translate)
        else:
            # Convert input to Path for file/directory modes
            input_path = Path(args.input)

            # Generate output_file if not specified
            output_file = args.output_file
            if output_file is None:
                # Auto-generate output filename based on input
                if input_path.is_dir():
                    # Batch mode: use directory name, save alongside directory
                    base_name = input_path.name or "transcription"
                    output_file = input_path.parent / f"{base_name}.txt"
                else:
                    # Single file mode: use input filename (same directory)
                    output_file = input_path.with_suffix(".txt")

            if input_path.is_dir():
                # Batch mode
                transcribe_batch(input_path, output_file, args.language, args.model, args.translate)
            else:
                # Single file mode
                transcribe_audio(input_path, output_file, args.language, args.model, args.translate)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user[/yellow]")
        console.print("[dim]Exiting gracefully...[/dim]\n")
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
