use batata_ai::local::model::resolve_device;
use batata_ai::local::whisper::{WhisperModel, WhisperSize};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: whisper_transcribe <audio.wav> [tiny|base|small|medium|large]");
        println!();
        println!("Supported: 16-bit PCM WAV files (mono or stereo, any sample rate)");
        println!("First run downloads the Whisper model from HuggingFace Hub.");
        return Ok(());
    }

    let audio_path = &args[1];
    let size = match args.get(2).map(|s| s.as_str()) {
        Some("tiny") => WhisperSize::Tiny,
        Some("small") => WhisperSize::Small,
        Some("medium") => WhisperSize::Medium,
        Some("large") => WhisperSize::Large,
        _ => WhisperSize::Base,
    };

    println!("=== batata-ai Whisper transcription ===");
    println!("Model: {:?}", size);
    println!("File:  {audio_path}\n");

    let device = resolve_device(false)?;
    let mut whisper = WhisperModel::download_and_load(size, &device)?;

    let result = whisper.transcribe_file(std::path::Path::new(audio_path))?;

    println!("Transcription:\n{}\n", result.text);

    if let Some(lang) = &result.language {
        println!("[language: {lang}]");
    }

    Ok(())
}
