import os
import io
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

env_path = "Key.env"
load_dotenv(dotenv_path=env_path)

apikey = os.getenv("API_KEY")
if not apikey:
    raise ValueError("Error | Could not find your Api Key inside the Online Key.env file layout")

client = genai.Client(api_key=apikey)

# Text Generation

def Text(prompt, instruction):
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=instruction)
        )
        return response.text
    except Exception as e:
        return f"Error = {e}"

# Image Generation

def Picture(prompt, format_type):
    format_clean = format_type.lower().strip()
    mime_type = "image/jpeg" if format_clean in ["jpeg", "jpg"] else "image/png"
    try:
        response = client.models.generate_content(
            model="gemini-nano-banana-2.1",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1", output_mime_type=mime_type)
            )
        )
        for part in response.candidates.content.parts:
            if part.inline_data:
                return Image.open(io.BytesIO(part.inline_data.data))
        return "Error = No Image Content Blocks were Returned"
    except Exception as e:
        return f"Error = {e}"

# Text to Speech Converter

def Text_To_Speech(text_prompt):
    tts_rules = "You are an expert voice actor. Create an expressive narrative transcript for reading this text out loud, including performance notes."
    return Text(text_prompt, tts_rules)

# Text Translation

def Translator_Text(text_to_translate, source_lang, target_lang):
    translation_instruction = f"Translate this text cleanly from {source_lang} to {target_lang}. Return ONLY the exact translation, with no conversational filler."
    return Text(text_to_translate, translation_instruction)

# Text Classification

def Classify_Text(text_to_classify):
    classifier_instruction = "Analyze this text and reply with exactly one word: either POSITIVE or NEGATIVE. Do not write full paragraphs."
    result = Text(text_to_classify, classifier_instruction)
    return f"Classification Analysis Output : {result.strip().title()}"
