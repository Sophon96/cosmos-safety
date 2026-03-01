# Unsloth must be imported before transformers for optimizations
# from unsloth import FastVisionModel
import torch
import transformers
import time

model_name = "nvidia/Cosmos-Reason2-2B"
# model, processor = FastVisionModel.from_pretrained(
#     model_name=model_name,
#     max_seq_length=4096,
#     load_in_4bit=False,  # Use 16-bit for full quality; set True for less VRAM
#     load_in_16bit=True,
# )
# FastVisionModel.for_inference(model)

model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)

processor: transformers.Qwen3VLProcessor = transformers.AutoProcessor.from_pretrained(model_name)

with open("prompt.txt", "r") as f:
    prompt = f.read()

video_messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [
            {
                "type": "video", 
                "video": "videos/snapshots/91ed530147ea8f380b10181c4e568865fa0e0996/output_phone/episode_049.mp4",
                "fps": 4,
            },
            {
                "type": "video", 
                "video": "videos/snapshots/91ed530147ea8f380b10181c4e568865fa0e0996/output/episode_049.mp4",
                "fps": 4,
            },

            {"type": "text", "text": (
                    "what is happening in these two videos?"
                )
            },
        ]
    },
]
start = time.time()
# Process inputs
inputs = processor.apply_chat_template(
    video_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    fps=4,
)

preprocess_time = time.time()

print("Time to preprocess: " + str(preprocess_time - start))
inputs = inputs.to(model.device)

# Run inference
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(output_text)
print("Total time: " + str(time.time() - start))
