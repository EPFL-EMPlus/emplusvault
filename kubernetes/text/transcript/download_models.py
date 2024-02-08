import whisperx

device = "cuda"
compute_type = "float16"
lang = "fr"

whisperx.load_model("large-v2", device, device_index=1, compute_type=compute_type, language=lang)
whisperx.load_align_model(language_code=lang, device=device)