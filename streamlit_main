import streamlit as st
from datasets import load_dataset
from diffusers import FluxPipeline
import ast
import google.generativeai as genai
import torch
import soundfile as sf
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# Initialize Google Generative AI
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your actual API key

def initialize_models():
    # Initialize the SpeechT5 models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return processor, model, vocoder, pipe

def generate_prompts_and_subtitles(topic):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Create an engaging story that explains {topic} using superheroes or other fictional characters. The story should be educational and easy for students to understand. Break down the story into frames. Each frame should be a dictionary with two keys: 'image_prompt' for the image description and 'narrator' for the accompanying text. Return the response as a list of dictionaries, like this: 
    [
        {{'image_prompt': 'description of image 1', 'narrator': 'text for image 1'}},
        {{'image_prompt': 'description of image 2', 'narrator': 'text for image 2'}}
    ]
    At the end, make a strong connection to the actual concept of {topic} and conclude the story effectively. Also make scientific and theoretical explanations where needed and make narrations very detailed. No additional text or commentary is needed.
    """    
    response = model.generate_content(prompt)
    story_and_prompts = ast.literal_eval(response)  # Ensure to parse the response correctly
    image_prompts = [frame['image_prompt'] for frame in story_and_prompts]
    narrators = [frame['narrator'] for frame in story_and_prompts]
    return image_prompts, narrators

def generate_images(pipe, image_prompts):
    for i in range(len(image_prompts)):
        image = pipe(
            image_prompts[i],
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=128,
            generator=torch.Generator("cuda").manual_seed(0)
        ).images[0]
        image.save(f"flux-schnell_{i}.png")

def generate_audio(processor, model, vocoder, story):
    for i in range(len(story)):
        inputs = processor(text=story[i], return_tensors="pt")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        sf.write(f"speech{i}.wav", speech.numpy(), samplerate=16000)

def create_video_from_images_and_audio(length, fps=12):
    clips = []
    for i in range(length):
        # Create the image and audio clips
        image_clip = ImageClip(f"flux-schnell_{i}.png").resize(0.5).set_duration(AudioFileClip(f"speech{i}.wav").duration).set_fps(fps)
        audio_clip = AudioFileClip(f"speech{i}.wav")
        
        # Combine image and audio
        image_clip = image_clip.set_audio(audio_clip)
        clips.append(image_clip)
    
    # Concatenate all video clips
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Write the final video
    final_video.write_videofile("final_video_no_subtitles.mp4", fps=fps, codec='libx264', preset="ultrafast")

def main():
    st.title("Educational Video Generator")
    
    # Input for the topic
    topic = st.text_input("Enter a topic you want to learn about:")
    
    if st.button("Generate Video"):
        if topic:
            with st.spinner("Generating video..."):
                processor, model, vocoder, pipe = initialize_models()
                image_prompts, story = generate_prompts_and_subtitles(topic)
                generate_images(pipe, image_prompts)
                generate_audio(processor, model, vocoder, story)
                create_video_from_images_and_audio(len(image_prompts))
            
            st.success("Video generated successfully!")
            st.video("final_video_no_subtitles.mp4")  # Display the generated video
        else:
            st.error("Please enter a topic.")

if __name__ == "__main__":
    main()
