# ---------------------------------------
# 🎥 YouTube Video Summarizer (Transcript + Whisper + Summarizer)
# ---------------------------------------

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # Fix SSL issues on macOS

import os
import re
import tempfile
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
import whisper
from transformers import pipeline

# ---------------------------------------
# 🧠 Load Summarization Model
# ---------------------------------------
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ---------------------------------------
# 🎧 Load Whisper Model (base model is faster)
# ---------------------------------------
whisper_model = whisper.load_model("base")

# ---------------------------------------
# 📜 Step 1: Try Transcript API or Captions
# ---------------------------------------
def extract_transcript(video_url):
    """
    Extracts transcript via youtube_transcript_api, or pytube captions if possible.
    Returns formatted text or a warning string starting with ⚠️
    """
    try:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
        if not video_id_match:
            return "⚠️ Invalid YouTube URL. Please enter a valid one."
        video_id = video_id_match.group(1)

        # 1️⃣ Try English transcript via youtube_transcript_api
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            formatter = TextFormatter()
            return formatter.format_transcript(transcript)
        except Exception:
            pass

        # 2️⃣ Try any other language available
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in transcripts:
                try:
                    fetched = t.fetch()
                    formatter = TextFormatter()
                    return formatter.format_transcript(fetched)
                except:
                    continue
        except Exception:
            pass

        # 3️⃣ Try pytube captions
        try:
            yt = YouTube(video_url)
            caption_srt = None
            for cap in yt.captions:
                code = getattr(cap, "code", "")
                if code and (code.startswith("en") or "english" in code.lower()):
                    caption_srt = cap.generate_srt_captions()
                    break

            if caption_srt:
                text = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> .*?\n", "", caption_srt)
                text = re.sub(r"\n+", " ", text).strip()
                return text
        except Exception:
            pass

        return "⚠️ No transcript or captions found for this video."

    except Exception as e:
        return f"⚠️ Error fetching transcript: {str(e)}"

# ---------------------------------------
# 🔊 Step 2: Whisper Audio Transcription
# ---------------------------------------
def transcribe_audio_with_whisper(video_url):
    """
    Robust audio downloader + Whisper transcription using yt_dlp.
    Handles all YouTube formats reliably.
    """
    import yt_dlp
    import tempfile
    import os
    import glob

    try:
        temp_dir = tempfile.mkdtemp()

        # Use a template so we can find the actual file later
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        # Download using yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Find the downloaded audio file
        audio_files = glob.glob(os.path.join(temp_dir, "*.mp3"))
        if not audio_files:
            # fallback: try any audio format
            audio_files = glob.glob(os.path.join(temp_dir, "*.m4a")) + \
                          glob.glob(os.path.join(temp_dir, "*.webm"))

        if not audio_files:
            return "⚠️ Audio download failed — no valid audio file found."

        audio_path = audio_files[0]

        # Double-check size
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return "⚠️ Audio download failed — file is empty or too small."

        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_path)
        text = result.get("text", "").strip()

        if not text:
            return "⚠️ Whisper could not detect any speech in the video."
        return text

    except Exception as e:
        return f"⚠️ Whisper transcription failed: {str(e)}"




# ---------------------------------------
# 🧩 Step 3: Summarization
# ---------------------------------------
def summarize_text(input_text):
    try:
        if len(input_text.strip()) == 0:
            return "⚠️ No text available to summarize."

        max_chunk_size = 1000
        sentences = input_text.split(". ")
        current_chunk = ""
        chunks = []

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        chunks.append(current_chunk.strip())

        summaries = []
        for chunk in chunks:
            summary = text_summary(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        return " ".join(summaries)
    except Exception as e:
        return f"⚠️ Error during summarization: {str(e)}"

# ---------------------------------------
# 🧠 Step 4: Combined Pipeline
# ---------------------------------------
def summarize_youtube_video(video_url):
    transcript_text = extract_transcript(video_url)

    # If no transcript found, use Whisper fallback
    if transcript_text.startswith("⚠️"):
        transcript_text = transcribe_audio_with_whisper(video_url)

    if transcript_text.startswith("⚠️"):
        return transcript_text

    summary_text = summarize_text(transcript_text)
    return summary_text

# ---------------------------------------
# 🎨 Gradio UI
# ---------------------------------------
gr.close_all()

demo = gr.Interface(
    fn=summarize_youtube_video,
    inputs=gr.Textbox(label="Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=abc123xyz"),
    outputs=gr.Textbox(label="Summarized Output", lines=10),
    title="🎬 YouTube Video Summarizer ",
    description="Summarizes any YouTube video using transcripts or speech-to-text with Whisper 🎧🤖"
)

demo.launch(server_name="0.0.0.0", server_port=8080)

