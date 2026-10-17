import io
from PIL import Image
from transformers import pipeline

# Text Generation

def Text(prompt, instruction, generator_pipeline):
    try:
        messages = [
            {"role": "system", "content": instruction if instruction else "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        outputs = generator_pipeline(messages, max_new_tokens=512, do_sample=True, temperature=0.7)
        
        if isinstance(outputs, list) and len(outputs) > 0:
            item = outputs[0]
            if isinstance(item, dict) and 'generated_text' in item:
                raw_text = item['generated_text']
                if isinstance(raw_text, list) and len(raw_text) > 0:
                    return raw_text[-1]['content'].strip()
                return str(raw_text).strip()
            return str(item).strip()
        elif isinstance(outputs, dict) and 'generated_text' in outputs:
            return str(outputs['generated_text']).strip()
            
        return str(outputs).strip()
    except Exception as e:
        return f"Error = {e}"

# Image Generation

def Picture(prompt, format_type):
    from PIL import Image, ImageDraw
    try:
        img = Image.new("RGB", (512, 512), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (502, 502)], outline=(0, 255, 128), width=4)
        draw.text((150, 240), f"Placeholder Image :\n{str(prompt).title()}", fill=(255, 255, 255))
        return img
    except Exception as e:
        return f"Error = {e}"

# Text to Speech Converter

def Text_To_Speech(text_prompt, tts_pipeline):
    try:

        result = tts_pipeline(text_prompt)
        
        byte_io = io.BytesIO()
        import soundfile as sf
        sf.write(byte_io, result["audio"], result["sampling_rate"], format='WAV')
        return byte_io.getvalue()
    except Exception as e:
        return f"Error = {e}"

# Text Translation

def Translator_Text(text_to_translate, source_lang, target_lang, generator_pipeline):
    translation_instruction = f"Translate this text cleanly from {source_lang} to {target_lang}. Return ONLY the exact translation, with no conversational filler or introduction text."
    return Text(text_to_translate, translation_instruction, generator_pipeline)

# Text Classification

def Classify_Text(text_to_classify, classifier_pipeline):
    try:
        result = classifier_pipeline(text_to_classify)
        if isinstance(result, list) and len(result) > 0:
            label = result[0]['label']
            return f"Classification Analysis Output : {label.strip().title()}"
        return f"Unexpected Pipeline Result Format : {result}"
    except Exception as e:
        return f"Error = {e}"
