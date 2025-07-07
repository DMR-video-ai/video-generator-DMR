import streamlit as st
import os
import tempfile
import shutil
from openai import OpenAI
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, CompositeVideoClip, TextClip
import requests
from elevenlabs import generate, set_api_key

st.set_page_config(page_title="DMR AI Video Generator", layout="wide")

# --- Setup API Keys ---
openai_api_key = st.secrets.get("OPENAI_API_KEY")
elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")
voice_id = st.secrets.get("VOICE_ID")

if not openai_api_key or not elevenlabs_api_key or not voice_id:
    st.error("Please add OPENAI_API_KEY, ELEVENLABS_API_KEY, and VOICE_ID to your Streamlit secrets.")
    st.stop()

set_api_key(elevenlabs_api_key)
client = OpenAI(api_key=openai_api_key)

# --- Helper Functions ---

def generate_scene_images_and_audio(prompt_text, temp_dir):
    # Use GPT-4 to break text into scenes
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Break the following text into 3 short scenes for a video."},
            {"role": "user", "content": prompt_text}
        ],
    )
    scenes_text = response.choices[0].message.content.strip().split('\n')
    if len(scenes_text) < 3:
        scenes_text = [prompt_text]  # fallback
    
    image_files = []
    audio_files = []
    subtitles = []

    for i, scene_text in enumerate(scenes_text):
        # Generate image with DALL·E 3
        img_resp = client.images.generate(
            model="dall-e-3",
            prompt=scene_text,
            size="512x512",
            n=1
        )
        img_url = img_resp.data[0].url
        img_data = requests.get(img_url).content
        img_path = os.path.join(temp_dir, f"scene_{i}.png")
        with open(img_path, "wb") as f:
            f.write(img_data)
        image_files.append(img_path)
        
        # Generate speech audio with ElevenLabs
        audio_data = generate(
            text=scene_text,
            voice=voice_id,
            model="eleven_multilingual_v1"
        )
        audio_path = os.path.join(temp_dir, f"scene_{i}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        audio_files.append(audio_path)
        
        subtitles.append(scene_text)
    return image_files, audio_files, subtitles

def create_video(image_files, audio_files, subtitles, output_path, music_path=None):
    clips = []
    for img, audio, subtitle in zip(image_files, audio_files, subtitles):
        audio_clip = AudioFileClip(audio)
        img_clip = ImageClip(img).set_duration(audio_clip.duration)
        
        # Create subtitle TextClip
        subtitle_clip = TextClip(subtitle, fontsize=24, color='white', bg_color='black', method='caption', size=(img_clip.w * 0.9, None))
        subtitle_clip = subtitle_clip.set_position(("center", "bottom")).set_duration(audio_clip.duration)
        
        video_clip = CompositeVideoClip([img_clip, subtitle_clip])
        video_clip = video_clip.set_audio(audio_clip)
        clips.append(video_clip)

    final_clip = concatenate_videoclips(clips)

    # Add background music if provided
    if music_path and os.path.exists(music_path):
        music_clip = AudioFileClip(music_path).volumex(0.1).loop(duration=final_clip.duration)
        final_audio = CompositeAudioClip([final_clip.audio, music_clip])
        final_clip = final_clip.set_audio(final_audio)
    
    final_clip.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    final_clip.close()

# --- Streamlit UI ---
st.title("🎥 DMR AI Video Generator")

text_input = st.text_area("Enter your story or script (3-4 sentences recommended):", height=150)

music_upload = st.file_uploader("Optional: Upload background music (mp3)", type=["mp3"])

if st.button("Generate Video") and text_input.strip():
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info("Generating scenes, images, and audio...")
        try:
            image_files, audio_files, subtitles = generate_scene_images_and_audio(text_input, temp_dir)

            music_path = None
            if music_upload:
                music_path = os.path.join(temp_dir, "background_music.mp3")
                with open(music_path, "wb") as f:
                    f.write(music_upload.read())

            output_video_path = os.path.join(temp_dir, "output_video.mp4")

            st.info("Creating video...")
            create_video(image_files, audio_files, subtitles, output_video_path, music_path)

            st.video(output_video_path)
            st.success("Video generated successfully!")

        except Exception as e:
            st.error(f"Error: {e}")
