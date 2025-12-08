# **Vision-to-VibeVoice-en**

> A Gradio-based demo for end-to-end vision-to-speech inference: Extract text or descriptions from images using Qwen2.5-VL-7B-Instruct, then convert to natural speech audio via Microsoft VibeVoice-Realtime-0.5B. Supports customizable queries, voice presets, and fidelity controls for applications like image captioning, document reading, or accessibility tools.

> Demo: https://huggingface.co/spaces/prithivMLmods/Vision-to-VibeVoice-en

## Features

- **Vision Processing**: Upload images and query for content analysis (e.g., "Caption the image" or "Read this page") using Qwen2.5-VL for accurate text extraction or description.
- **Text-to-Speech Generation**: Converts extracted text to high-fidelity audio with VibeVoice, supporting multiple speaker voices from presets.
- **Voice Customization**: Select from available voice files; adjust CFG scale (1.0-3.0) for speech naturalness.
- **Advanced Controls**: Tune max tokens (up to 4096) and temperature for vision output; editable text for post-processing.
- **Progress and Logging**: Real-time progress bar and status updates; saves audio as WAV files in `./outputs`.
- **Custom Theme**: OrangeRedTheme with gradients for an engaging UI.
- **Examples and Queueing**: Pre-loaded examples; handles up to 40 queued inferences.
- **Error Handling**: Graceful fallbacks for missing voices or generation failures; console warnings for setup issues.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (required for float16; falls back to CPU but slower).
- Git for cloning repositories.
- Hugging Face account (optional, for model caching via `huggingface_hub`).
- VibeVoice repository structure (cloned via requirements for custom modules).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Vision-to-VibeVoice-en.git
   cd Vision-to-VibeVoice-en
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   git+https://github.com/microsoft/VibeVoice.git
   git+https://github.com/huggingface/peft.git
   transformers-stream-generator
   huggingface_hub
   qwen-vl-utils
   pyvips-binary
   sentencepiece
   opencv-python
   docling-core
   transformers
   python-docx
   torchvision
   supervision 
   matplotlib
   accelerate
   pdf2image
   reportlab
   markdown
   requests
   librosa
   pymupdf
   hf_xet
   spaces
   pyvips
   pillow
   gradio
   einops
   httpx
   torch
   fpdf
   timm
   av
   ```

3. Ensure voice files are in `demo/voices/streaming_model/` (auto-detected from VibeVoice repo).

4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Upload Image**: Drag-and-drop or use the example (e.g., a document or scene).

2. **Set Query**: Enter a prompt like "Give a short description" or "Extract all text." Default: Safety check description.

3. **Configure Voice**: Select a speaker from the dropdown (e.g., from presets like "default" or custom names).

4. **Adjust Settings**:
   - CFG Scale: Higher values (e.g., 2.0) for more expressive speech.
   - Expand "Advanced Options" for token limits and temperature.

5. **Generate**: Click "Generate Speech." Monitor progress; results show extracted text (editable), audio player, and status.

6. **Output**:
   - Text: Vision model response.
   - Audio: Playable WAV; downloadable from `./outputs`.
   - Status: Success message or error details.

### Example Workflow
- Upload a book page image.
- Query: "Read the visible text aloud."
- Select "English" voice, CFG=1.5.
- Output: Extracted paragraph as narrated audio.

## Troubleshooting

- **VibeVoice Modules Missing**: Ensure `git+https://github.com/microsoft/VibeVoice.git` clones correctly; check `demo/voices/streaming_model/` for `.pt` files.
- **CUDA Errors**: Verify `torch.cuda.is_available()`; use `torch.float32` if float16 fails. Clear cache with `torch.cuda.empty_cache()`.
- **No Voices Found**: Directory warning in console; add custom `.pt` files or fallback to default.
- **Generation Fails**: Reduce max tokens for long images; check for OOM with `nvidia-smi`.
- **Audio Silent**: Ensure sample rate (24000 Hz); test TTS standalone.
- **UI Issues**: Set `ssr_mode=True` if rendering fails; increase `max_size` for queues.
- **Transformers Version**: Locked to 4.57.3; update via `check_and_install_package` if needed.

## Contributing

Contributions welcome! Fork the repo, branch for features (e.g., more voices, multilingual support), and submit PRs with tests. Ideas:
- Batch processing for multiple images.
- Integration with real-time video.
- Custom voice uploads.

Repository: [https://github.com/PRITHIVSAKTHIUR/Vision-to-VibeVoice-en.git](https://github.com/PRITHIVSAKTHIUR/Vision-to-VibeVoice-en.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
