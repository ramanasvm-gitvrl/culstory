import os
import json
import re
import subprocess
from typing import List, Optional
import requests
from gtts import gTTS
from PIL import Image
import streamlit as st
from openai import OpenAI
from mutagen.mp3 import MP3
from dotenv import load_dotenv
load_dotenv()

# ---------- CONFIG ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- STREAMLIT PAGE SETTINGS ----------
st.set_page_config(
    page_title="🎬 AI Story Video Generator",
    layout="wide",
    page_icon="🎥",
)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Choose narration language and illustration style.")

languages = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Gujarati (ગુજરાતી)": "gu",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
}

selected_lang = st.sidebar.selectbox("🎙️ Narration Language", list(languages.keys()), index=0)
theme_style = st.sidebar.selectbox("🎨 Illustration Style", ["children's storybook", "realistic", "fantasy", "cartoon"])
st.sidebar.info("💡 Tip: Try prompts like *‘Tenali Rama story of 100 words’* or *‘Akbar and Birbal moral story’*")

# ---------- TITLE ----------
st.markdown(
    "<h1 style='text-align:center;'>🎬 AI Story Video Generator</h1>"
    "<p style='text-align:center;'>Generate a narrated illustrated story video in your chosen language.</p>",
    unsafe_allow_html=True
)

# ---------- STEP 1: USER INPUT ----------
prompt = st.text_area("✏️ Enter your story idea or prompt:", placeholder="e.g. Tenali Rama story of 100 words", height=100)

generate_btn = st.button("🚀 Generate Story Video")

# ---------- CORE FUNCTIONS ----------
def generate_story_from_prompt(prompt: str) -> str:
    """Generate story text using OpenAI GPT-4."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a creative story writer for children."},
                {"role": "user", "content": f"Write a 100-150 word short story for children based on: {prompt}"}
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ Story generation failed: {e}")
        return None


def extract_keywords(story_text: str) -> List[str]:
    """Extract main keywords (characters, settings, themes) with fallback."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract 5-7 important visual keywords (characters, locations, objects) from the story. Respond with a JSON list."},
                {"role": "user", "content": story_text},
            ],
            max_tokens=150,
        )
        raw_output = response.choices[0].message.content.strip()

        # Clean "undefined" or extra junk
        cleaned = raw_output.replace("undefined", "").strip("`").strip()
        st.write("🔎 Raw keyword output (from model)", cleaned)

        # Try JSON first
        try:
            keywords = json.loads(cleaned)
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords if k.strip()]
        except json.JSONDecodeError:
            pass  # fallback to regex if not valid JSON

        # Regex fallback
        keywords = re.findall(r'"(.*?)"', cleaned)
        return [kw for kw in keywords if kw.strip()]
    except Exception as e:
        st.error(f"Keyword extraction failed: {e}")
        return []


def fetch_image_pollinations_for_keyword(keyword: str, theme_prompt: str, save_as: str) -> Optional[str]:
    """Fetch an image for the given keyword using Pollinations API."""
    try:
        url = f"https://image.pollinations.ai/prompt/{keyword} - {theme_prompt}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_as, "wb") as f:
                f.write(response.content)
            return save_as
        else:
            st.warning(f"⚠️ Pollinations returned status {response.status_code} for '{keyword}'")
            return None
    except Exception as e:
        st.warning(f"Error fetching image for {keyword}: {e}")
        return None


def generate_audio(text: str, language_code: str) -> Optional[str]:
    """Generate audio narration using gTTS."""
    try:
        audio_path = "story_audio.mp3"
        tts = gTTS(text=text, lang=language_code)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"⚠️ Audio generation failed: {e}")
        return None


def get_audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds using mutagen (works on Streamlit Cloud)."""
    try:
        audio = MP3(audio_path)
        return audio.info.length
    except Exception as e:
        st.warning(f"⚠️ Could not determine audio duration using mutagen: {e}")
        return None



def create_slideshow(image_files: List[str], audio_file: str, output_file: str = "story_video.mp4") -> Optional[str]:
    """Combine images and narration into a video using ffmpeg."""
    try:
        audio_duration = get_audio_duration(audio_file)
        if not audio_duration:
            raise ValueError("Could not determine audio duration")

        img_duration = audio_duration / len(image_files)
        input_txt = "inputs.txt"
        with open(input_txt, "w") as f:
            for img in image_files:
                f.write(f"file '{img}'\n")
                f.write(f"duration {img_duration}\n")
            f.write(f"file '{image_files[-1]}'\n")

        # Capture ffmpeg output to debug
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", input_txt,
            "-i", audio_file, "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p", "-shortest", output_file
        ], capture_output=True, text=True)

        if result.returncode != 0:
            st.error(f"❌ Slideshow creation failed:\n{result.stderr}")
            return None

        return output_file
    except Exception as e:
        st.error(f"❌ Slideshow creation failed: {e}")
        return None

# ---------- MAIN WORKFLOW ----------
if generate_btn and prompt:
    with st.spinner("✨ Generating your story..."):
        story = generate_story_from_prompt(prompt)

    if story:
        with st.expander("📖 Story", expanded=True):
            st.markdown(f"**Generated Story in {selected_lang}:**")
            st.write(story)

        with st.spinner("🧩 Extracting keywords..."):
            keywords = extract_keywords(story)

        if keywords:
            with st.expander("🔑 Extracted Keywords", expanded=True):
                st.success("✅ Keywords extracted successfully!")
                st.write(keywords)

            # IMAGE GENERATION
            with st.expander("🎨 Generating Illustrations", expanded=True):
                image_files = []
                progress_bar = st.progress(0)
                for i, keyword in enumerate(keywords):
                    img_path = f"image_{i}.jpg"
                    result = fetch_image_pollinations_for_keyword(keyword, theme_style, img_path)
                    if result:
                        image_files.append(img_path)
                    progress_bar.progress((i + 1) / len(keywords))
                if image_files:
                    st.image(image_files, width=150, caption=keywords)
                else:
                    st.warning("⚠️ No images could be fetched.")

            # AUDIO GENERATION
            with st.expander("🎙️ Narration", expanded=True):
                st.info(f"Generating narration in **{selected_lang}**...")
                audio_file = generate_audio(story, languages[selected_lang])
                if audio_file:
                    st.audio(audio_file)
                else:
                    st.error("❌ Audio generation failed.")

            # VIDEO CREATION
            with st.expander("🎞️ Final Story Video", expanded=True):
                if image_files and audio_file:
                    st.info("Creating final video slideshow...")
                    video_path = create_slideshow(image_files, audio_file)
                    if video_path and os.path.exists(video_path):
                        st.success("✅ Video generated successfully!")
                        st.video(video_path)
                    else:
                        st.error("❌ Failed to create video.")
