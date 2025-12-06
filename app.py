import os
import sys
import cv2
import time
import copy
import random
import torch
import spaces
import requests
import subprocess
import gradio as gr
from PIL import Image
import importlib.util
from threading import Thread
from typing import Iterable, Optional, Tuple, List

def check_and_install_package(package_name, import_name=None, pip_name=None):
    """Check if a package is installed, and if not, install it."""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name

    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
        print(f"✓ {package_name} installed successfully")
    return True

print("Checking and installing transformers==4.57.3 ...")
check_and_install_package("transformers", "transformers", "transformers==4.57.3")
print("Done!")

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)

try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
except ImportError:
    print("CRITICAL WARNING: 'vibevoice' modules not found. Ensure the vibevoice repository structure is present.")
    VibeVoiceStreamingForConditionalGenerationInference = None
    VibeVoiceStreamingProcessor = None

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
.generating {
    border: 2px solid #4682B4;
}
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using Main Device: {device}")

QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Loading OCR Model: {QWEN_VL_MODEL_ID}...")

qwen_processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_ID, trust_remote_code=True)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_VL_MODEL_ID,
    #attn_implementation="flash_attention_2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

print("Model loaded successfully.")

TTS_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
print(f"Loading TTS Model: {TTS_MODEL_PATH}...")
print("VibeVoice Model loaded successfully.")

tts_processor = VibeVoiceStreamingProcessor.from_pretrained(TTS_MODEL_PATH)

tts_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    TTS_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
)
tts_model.eval()
tts_model.set_ddpm_inference_steps(num_steps=5)

class VoiceMapper:
    """Maps speaker names to voice file paths"""
    def __init__(self):
        self.setup_voice_presets()
        new_dict = {}
        for name, path in self.voice_presets.items():
            if "_" in name: name = name.split("_")[0]
            if "-" in name: name = name.split("-")[-1]
            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "demo/voices/streaming_model")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        self.voice_presets = {}
        pt_files = [f for f in os.listdir(voices_dir) if f.lower().endswith(".pt") and os.path.isfile(os.path.join(voices_dir, f))]

        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(voices_dir, pt_file)
            self.voice_presets[name] = full_path

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {name: path for name, path in self.voice_presets.items() if os.path.exists(path)}
        print(f"Found {len(self.available_voices)} voice files.")

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        if self.voice_presets:
            return list(self.voice_presets.values())[0]
        return ""

VOICE_MAPPER = VoiceMapper()
print("TTS Model loaded successfully.")

@spaces.GPU
def process_pipeline(
    image: Image.Image,
    query: str,
    speaker_name: str,
    cfg_scale: float,
    ocr_max_tokens: int,
    ocr_temp: float,
    progress=gr.Progress()
):
    """
    Combined pipeline: Image - Text -> TTS - Audio
    """
    if image is None:
        return "Please upload an image.", None, "Error: No image provided."
    
    progress(0.2, desc="Analyzing Image ()...")
    
    if not query: 
        query = "Analyze the content perfectly."
        
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": query},
        ]
    }]
    
    prompt_full = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = qwen_processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    generated_ids = qwen_model.generate(
        **inputs,
        max_new_tokens=ocr_max_tokens,
        do_sample=True,
        temperature=ocr_temp,
        top_p=0.9,
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    extracted_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    extracted_text = extracted_text.replace("<|im_end|>", "").strip()
    
    progress(0.5, desc=f"Analyzing completed. Converting to speech ({len(extracted_text)} chars)...")
    
    if not extracted_text:
        return extracted_text, None, "produced no text."
        
    try:
        full_script = extracted_text.replace("'", "'").replace('"', '"').replace('"', '"')
        
        voice_path = VOICE_MAPPER.get_voice_path(speaker_name)
        if not voice_path:
             return extracted_text, None, "Error: Voice file not found."

        all_prefilled_outputs = torch.load(voice_path, map_location="cuda", weights_only=False)
        
        tts_inputs = tts_processor.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        tts_model.to("cuda")
        for k, v in tts_inputs.items():
            if torch.is_tensor(v):
                tts_inputs[k] = v.to("cuda")
                
        with torch.cuda.amp.autocast():
            outputs = tts_model.generate(
                **tts_inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=tts_processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs)
            )
            
        tts_model.to("cpu")
        torch.cuda.empty_cache()
        
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            sample_rate = 24000
            
            output_dir = "./outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_{int(time.time())}.wav")
            
            tts_processor.save_audio(
                outputs.speech_outputs[0].cpu(),
                output_path=output_path,
            )
            
            status = f"✅ Success! Text Length: {len(extracted_text)} chars."
            return extracted_text, output_path, status
        else:
            return extracted_text, None, "TTS Generation failed (no output)."

    except Exception as e:
        tts_model.to("cpu")
        torch.cuda.empty_cache()
        import traceback
        return extracted_text, None, f"Error during TTS: {str(e)}"

url = "https://huggingface.co/datasets/strangervisionhf/image-examples/resolve/main/2.jpg?download=true"
example_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

with gr.Blocks() as demo:
    gr.Markdown("# **Vision-to-VibeVoice-en**", elem_id="main-title")
    gr.Markdown("Perform vision-to-audio inference with [Qwen2.5VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) + [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B).")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Vision Input")
            image_upload = gr.Image(type="pil", label="Upload Image", value=example_image, height=300)
            image_query = gr.Textbox(label="Enter the prompt", value="Give a short description indicating whether the image is safe or unsafe.", placeholder="E.g., Read this page...")
            
            gr.Markdown("### 2. Voice Settings")
            voice_choices = list(VOICE_MAPPER.available_voices.keys())
            if not voice_choices: voice_choices = ["Default"]
                
            speaker_dropdown = gr.Dropdown(
                 choices=voice_choices,
                value=voice_choices[0],
                label="Speaker Voice"
            )
                
            cfg_slider = gr.Slider(minimum=1.0, maximum=3.0, value=1.5, step=0.1, label="CFG Scale (Speech Fidelity)")
                
            with gr.Accordion("Advanced Options", open=False):
                max_new_tokens = gr.Slider(label="Max Tokens", minimum=128, maximum=4096, step=128, value=1024)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.1)

            submit_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Results", elem_id="output-title")
            
            text_output = gr.Textbox(
                label="Extracted Text (Editable)", 
                interactive=True, 
                lines=11, 
            )
            
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath",
                interactive=False
            )
            
            status_output = gr.Textbox(label="Status Log", lines=2)
            
            gr.Examples(
                examples=[["Caption the image...", "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/venice.jpg"]],
                inputs=[image_query, image_upload],
                label="Example"
            )

    submit_btn.click(
        fn=process_pipeline,
        inputs=[
            image_upload, 
            image_query, 
            speaker_dropdown, 
            cfg_slider,
            max_new_tokens,
            temperature
        ],
        outputs=[text_output, audio_output, status_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=40).launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)