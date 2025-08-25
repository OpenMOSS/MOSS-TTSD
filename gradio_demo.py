import gradio as gr
import torch
import torchaudio
import tempfile
import json
import os
from typing import Optional, Tuple

from generation_utils import load_model, process_batch

def load_examples_from_jsonl():
    """
    Load examples from examples/examples.jsonl and convert to format for both ROLE and SINGLE modes
    """
    jsonl_paths = ["examples/examples.jsonl", "examples/examples_single_reference.jsonl"]

    role_examples = []
    single_examples = []

    lines = []

    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found")
            continue
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)

    for line in lines:
        line = line.strip()
        if not line:
                continue

        data = json.loads(line)
        
        # Extract required fields
        text = data.get('text', '')
        base_path = data.get('base_path', 'examples')
        use_normalize = data.get('use_normalize', True)
        
        # Check if this is a role-based example (has speaker1 and speaker2 audio)
        if 'prompt_audio_speaker1' in data and 'prompt_audio_speaker2' in data:
            # Role mode example
            audio_mode = "Role"
            prompt_audio_1 = os.path.join(base_path, data['prompt_audio_speaker1'])
            prompt_text_1 = data.get('prompt_text_speaker1', '')
            prompt_audio_2 = os.path.join(base_path, data['prompt_audio_speaker2'])
            prompt_text_2 = data.get('prompt_text_speaker2', '')
            
            example = [text, audio_mode, prompt_audio_1, prompt_text_1, prompt_audio_2, prompt_text_2, use_normalize]
            role_examples.append(example)
            
        # Check if this is a single audio example (has prompt_audio and prompt_text)
        elif 'prompt_audio' in data and 'prompt_text' in data:
            # Single mode example
            audio_mode = "Single"
            prompt_audio = os.path.join(base_path, data['prompt_audio'])
            prompt_text = data.get('prompt_text', '')
            
            example = [text, audio_mode, prompt_audio, prompt_text, use_normalize]
            single_examples.append(example)
    
    print(f"Loaded {len(role_examples)} role examples and {len(single_examples)} single examples from {jsonl_paths}")
    return role_examples, single_examples

# Load examples from JSONL file
ROLE_EXAMPLES, SINGLE_EXAMPLES = load_examples_from_jsonl()

# Language configuration
LANGUAGES = {
    "English": {
        "title": "MOSS-TTSD🪐 Dialogue Generation",
        "script_input": "### Script Input",
        "text_to_synthesize": "Text to Synthesize",
        "text_placeholder": "Text to be synthesized, format: [S1]Role1 text[S2]Role2 text",
        "use_normalize": "Use text normalization",
        "normalize_info": "Recommended to enable, improves handling of numbers, punctuation, etc.",
        "audio_input_mode": "### Audio Input Mode",
        "select_input_mode": "Select input mode",
        "mode_info": "Single Audio: Upload one audio with [S1][S2] text; Role Audio: Upload separate audio for Role1 and Role2",
        "drag_drop_audio": "Drag and drop audio here - or - click to upload",
        "single_warning": "⚠️ Warning: The current model (v0.5) performs poorly with single-speaker reference audio. Please upload audio containing two speakers.",
        "prompt_text": "Prompt Text",
        "prompt_placeholder": "Format: [S1]Role1 text[S2]Role2 text",
        "role1_audio": "**Role1 Audio**",
        "role1_audio_file": "Role1 Audio File",
        "role1_text": "Role1 Text",
        "role1_placeholder": "Role1 text content",
        "role2_audio": "**Role2 Audio**",
        "role2_audio_file": "Role2 Audio File",
        "role2_text": "Role2 Text",
        "role2_placeholder": "Role2 text content",
        "generate_audio": "Generate Audio",
        "generated_audio": "Generated Audio",
        "status_info": "Status Information",
        "examples": "### Examples",
        "examples_desc": "Click on examples below to auto-fill the form",
        "role_examples": "Role Mode Examples",
        "single_examples": "Single Audio Mode Examples",
        "role_headers": ["Text to Synthesize", "Input Mode", "Role1 Audio File", "Role1 Text", "Role2 Audio File", "Role2 Text", "Use Normalize"],
        "single_headers": ["Text to Synthesize", "Input Mode", "Audio File", "Prompt Text", "Use Normalize"]
    },
    "中文": {
        "title": "MOSS-TTSD🪐 对话语音生成",
        "script_input": "### 文本输入",
        "text_to_synthesize": "要合成的文本",
        "text_placeholder": "要合成的文本，格式：[S1]角色1文本[S2]角色2文本",
        "use_normalize": "使用文本规范化",
        "normalize_info": "建议启用，改善数字、标点符号等的处理",
        "audio_input_mode": "### 音频输入模式",
        "select_input_mode": "选择输入模式",
        "mode_info": "单音频：上传一个包含[S1][S2]文本的音频；角色音频：分别为角色1和角色2上传音频",
        "drag_drop_audio": "拖拽音频文件到此处 - 或 - 点击上传",
        "single_warning": "⚠️ 警告：当前模型（v0.5）在单说话人参考音频上表现较差，请上传包含两个说话人的音频。",
        "prompt_text": "提示文本",
        "prompt_placeholder": "格式：[S1]角色1文本[S2]角色2文本",
        "role1_audio": "**角色1音频**",
        "role1_audio_file": "角色1音频文件",
        "role1_text": "角色1文本",
        "role1_placeholder": "角色1文本内容",
        "role2_audio": "**角色2音频**",
        "role2_audio_file": "角色2音频文件",
        "role2_text": "角色2文本",
        "role2_placeholder": "角色2文本内容",
        "generate_audio": "生成音频",
        "generated_audio": "生成的音频",
        "status_info": "状态信息",
        "examples": "### 示例",
        "examples_desc": "点击下方示例自动填充表单",
        "role_examples": "角色模式示例",
        "single_examples": "单音频模式示例",
        "role_headers": ["要合成的文本", "输入模式", "角色1音频文件", "角色1文本", "角色2音频文件", "角色2文本", "使用规范化"],
        "single_headers": ["要合成的文本", "输入模式", "音频文件", "提示文本", "使用规范化"]
    }
}

# Model configuration
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_32k_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8

# Global variables for caching loaded models
tokenizer = None
model = None
spt = None
device = None

def initialize_model():
    """Initialize model (load only on first call)"""
    global tokenizer, model, spt, device
    
    if tokenizer is None:
        print("Initializing model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model, spt = load_model(MODEL_PATH, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
        spt = spt.to(device)
        model = model.to(device)
        print("Model initialization completed!")
    
    return tokenizer, model, spt, device

def process_single_audio_generation(
    text_input: str,
    audio_mode: str,
    prompt_text_single: str,
    prompt_audio_single: Optional[str] = None,
    prompt_text_1: str = "",
    prompt_audio_1: Optional[str] = None,
    prompt_text_2: str = "",
    prompt_audio_2: Optional[str] = None,
    use_normalize: bool = True
) -> Tuple[Optional[str], str]:
    """
    Process single audio generation request
    
    Args:
        text_input: Text to synthesize
        prompt_text_single: Prompt text for single audio
        prompt_audio_single: Single audio file path
        prompt_text_1: Role1 text
        prompt_audio_1: Role1 audio file path
        prompt_text_2: Role2 text
        prompt_audio_2: Role2 audio file path
        use_normalize: Whether to use text normalization
    
    Returns:
        Generated audio file path and status information
    """
    try:
        # Initialize model
        tokenizer, model, spt, device = initialize_model()
        
        # Build input item
        item = {
            "text": text_input,
        }
        
        # Handle different audio input modes (mutually exclusive)
        if audio_mode == "Single":
            # Use single audio mode
            item["prompt_audio"] = prompt_audio_single
            item["prompt_text"] = prompt_text_single
        elif audio_mode == "Role" and prompt_audio_1 and prompt_audio_2:
            # Use role audio mode (requires both audio files)
            item["prompt_audio_speaker1"] = prompt_audio_1
            item["prompt_text_speaker1"] = prompt_text_1 if prompt_text_1 else ""
            item["prompt_audio_speaker2"] = prompt_audio_2
            item["prompt_text_speaker2"] = prompt_text_2 if prompt_text_2 else ""
        elif audio_mode == "Role" and prompt_audio_1:
            # Only Role 1 audio provided, treat as single audio
            print("Only Role 1 audio provided, treating as single audio.")
            item["prompt_audio"] = prompt_audio_1
            item["prompt_text"] = prompt_text_1 if prompt_text_1 else ""
        elif audio_mode == "Role" and prompt_audio_2:
            # Only Role 2 audio provided, treat as single audio
            print("Only Role 2 audio provided, treating as single audio.")
            item["prompt_audio"] = prompt_audio_2
            item["prompt_text"] = prompt_text_2 if prompt_text_2 else ""
        else:
            return None, "Error: Please select a mode and provide corresponding audio files\n- Single Audio Mode: Provide one audio file and corresponding text\n- Role Mode: Provide audio files for Role1 and Role2"
        
        # Set random seed to ensure reproducible results
        # import accelerate
        # accelerate.utils.set_seed(42)
        
        # Process batch (single item)
        actual_texts_data, audio_results = process_batch(
            batch_items=[item],
            tokenizer=tokenizer,
            model=model,
            spt=spt,
            device=device,
            system_prompt=SYSTEM_PROMPT,
            start_idx=0,
            use_normalize=use_normalize
        )
        
        # Check results
        if not audio_results or audio_results[0] is None:
            return None, "Error: Audio generation failed"
        
        audio_result = audio_results[0]
        
        # Create temporary output file
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        
        # Save audio
        torchaudio.save(output_path, audio_result["audio_data"], audio_result["sample_rate"])
        
        # Build status information (using English since this is server-side output)
        status_info = f"""
✅ Generation successful!
📊 Audio Information:
   - Sample Rate: {audio_result["sample_rate"]} Hz
   - Audio Length: {audio_result["audio_data"].shape[-1] / audio_result["sample_rate"]:.2f} seconds
   - Channels: {audio_result["audio_data"].shape[0]}

📝 Text Processing Information:
   - Original Text: {actual_texts_data[0]['original_text'][:100]}...
   - Final Text: {actual_texts_data[0]['final_text'][:100]}...
   - Use Normalize: {actual_texts_data[0]['use_normalize']}
        """
        
        return output_path, status_info
        
    except Exception as e:
        import traceback
        error_msg = f"Error: Audio generation failed: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
        return None, error_msg

# Create Gradio interface
def create_gradio_interface() -> gr.Blocks:
    with gr.Blocks(title="MOSS-TTSD🪐 Dialogue Generation", theme=gr.themes.Soft()) as demo:
        
        # Language selection at the top
        with gr.Row():
            language_selector = gr.Radio(
                choices=["English", "中文"],
                value="English",
                label="Language / 语言",
                info="Select interface language / 选择界面语言"
            )
        
        # Title and header (will be updated based on language)
        title_md = gr.Markdown("# MOSS-TTSD🪐 Dialogue Generation")
        github_md = gr.Markdown("### [Github](https://github.com/OpenMOSS/MOSS-TTSD)")
        
        with gr.Row():
            # Left input area
            with gr.Column(scale=1):
                script_input_md = gr.Markdown("### Script Input")
                
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Text to be synthesized, format: [S1]Role1 text[S2]Role2 text",
                    lines=6,
                )
                
                use_normalize_single = gr.Checkbox(
                    label="Use text normalization",
                    value=True,
                    info="Recommended to enable, improves handling of numbers, punctuation, etc."
                )
            
            # Right audio input area
            with gr.Column(scale=1):
                audio_input_mode_md = gr.Markdown("### Audio Input Mode")
                
                # Audio input mode selection
                audio_mode = gr.Radio(
                    choices=["Single", "Role"],
                    value="Single",
                    label="Select input mode",
                    info="Single Audio: Upload one audio with [S1][S2] text; Role Audio: Upload separate audio for Role1 and Role2"
                )
                
                # Single audio mode
                with gr.Group(visible=True) as single_mode_group:
                    single_warning_md = gr.Markdown(
                        "⚠️ Warning: The current model (v0.5) performs poorly with single-speaker reference audio. Please upload audio containing two speakers."
                    )
                    prompt_audio_single = gr.File(
                        label="Drag and drop audio here - or - click to upload",
                        file_types=["audio"],
                        type="filepath"
                    )
                    prompt_text_single = gr.Textbox(
                        label="Prompt Text",
                        placeholder="Format: [S1]Role1 text[S2]Role2 text",
                        lines=3,
                    )
                
                # Role audio mode
                with gr.Group(visible=False) as role_mode_group:
                    with gr.Row():
                        with gr.Column():
                            role1_audio_md = gr.Markdown("**Role1 Audio**")
                            prompt_audio_1 = gr.File(
                                label="Role1 Audio File",
                                file_types=["audio"],
                                type="filepath"
                            )
                            prompt_text_1 = gr.Textbox(
                                label="Role1 Text",
                                placeholder="Role1 text content",
                                lines=2
                            )
                        
                        with gr.Column():
                            role2_audio_md = gr.Markdown("**Role2 Audio**")
                            prompt_audio_2 = gr.File(
                                label="Role2 Audio File",
                                file_types=["audio"],
                                type="filepath"
                            )
                            prompt_text_2 = gr.Textbox(
                                label="Role2 Text",
                                placeholder="Role2 text content",
                                lines=2
                            )
        
        # Generate button
        with gr.Row():
            generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")
        
        # Output area
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio", type="filepath")
                status_info = gr.Textbox(
                    label="Status Information",
                    lines=10,
                    interactive=False
                )
        
        # Examples area
        with gr.Row():
            with gr.Column():
                examples_md = gr.Markdown("### Examples")
                examples_desc_md = gr.Markdown("Click on examples below to auto-fill the form")

                # Role mode examples
                with gr.Group():
                    role_examples_md = gr.Markdown("**Role Mode Examples**")
                    role_examples = gr.Examples(
                        examples=ROLE_EXAMPLES,
                        inputs=[text_input, audio_mode, prompt_audio_1, prompt_text_1, prompt_audio_2, prompt_text_2, use_normalize_single],
                    )
                
                # Single audio mode examples
                with gr.Group():
                    single_examples_md = gr.Markdown("**Single Audio Mode Examples**")
                    single_examples = gr.Examples(
                        examples=SINGLE_EXAMPLES,
                        inputs=[text_input, audio_mode, prompt_audio_single, prompt_text_single, use_normalize_single],
                    )
        
        # Event handlers
        
        # Language change event
        def update_language(lang):
            """Update interface language"""
            texts = LANGUAGES[lang]
            
            # Update demo title
            demo.title = texts["title"]
            
            return (
                gr.Markdown(f"# {texts['title']}"),  # title_md
                texts["script_input"],  # script_input_md
                gr.Textbox(
                    label=texts["text_to_synthesize"],
                    placeholder=texts["text_placeholder"],
                    lines=6,
                ),  # text_input
                gr.Checkbox(
                    label=texts["use_normalize"],
                    value=True,
                    info=texts["normalize_info"]
                ),  # use_normalize_single
                texts["audio_input_mode"],  # audio_input_mode_md
                gr.Radio(
                    choices=["Single", "Role"],
                    value="Single",
                    label=texts["select_input_mode"],
                    info=texts["mode_info"]
                ),  # audio_mode
                gr.Markdown(texts["single_warning"]),  # single_warning_md
                gr.File(
                    label=texts["drag_drop_audio"],
                    file_types=["audio"],
                    type="filepath"
                ),  # prompt_audio_single
                gr.Textbox(
                    label=texts["prompt_text"],
                    placeholder=texts["prompt_placeholder"],
                    lines=3,
                ),  # prompt_text_single
                texts["role1_audio"],  # role1_audio_md
                gr.File(
                    label=texts["role1_audio_file"],
                    file_types=["audio"],
                    type="filepath"
                ),  # prompt_audio_1
                gr.Textbox(
                    label=texts["role1_text"],
                    placeholder=texts["role1_placeholder"],
                    lines=2
                ),  # prompt_text_1
                texts["role2_audio"],  # role2_audio_md
                gr.File(
                    label=texts["role2_audio_file"],
                    file_types=["audio"],
                    type="filepath"
                ),  # prompt_audio_2
                gr.Textbox(
                    label=texts["role2_text"],
                    placeholder=texts["role2_placeholder"],
                    lines=2
                ),  # prompt_text_2
                gr.Button(texts["generate_audio"], variant="primary", size="lg"),  # generate_btn
                gr.Audio(label=texts["generated_audio"], type="filepath"),  # output_audio
                gr.Textbox(
                    label=texts["status_info"],
                    lines=10,
                    interactive=False
                ),  # status_info
                texts["examples"],  # examples_md
                texts["examples_desc"],  # examples_desc_md
                texts["role_examples"],  # role_examples_md
                texts["single_examples"],  # single_examples_md
                gr.Dataset(headers=texts["role_headers"]),
                gr.Dataset(headers=texts["single_headers"]),
            )
        
        language_selector.change(
            fn=update_language,
            inputs=[language_selector],
            outputs=[
                title_md, script_input_md, text_input, use_normalize_single,
                audio_input_mode_md, audio_mode, single_warning_md,
                prompt_audio_single, prompt_text_single,
                role1_audio_md, prompt_audio_1, prompt_text_1,
                role2_audio_md, prompt_audio_2, prompt_text_2,
                generate_btn, output_audio, status_info,
                examples_md, examples_desc_md, role_examples_md, single_examples_md,
                role_examples.dataset, single_examples.dataset
            ]
        )
        
        # Audio mode toggle event
        def toggle_audio_mode(mode):
            if mode == "Single":
                return gr.Group(visible=True), gr.Group(visible=False)
            else:
                return gr.Group(visible=False), gr.Group(visible=True)
        
        audio_mode.change(
            fn=toggle_audio_mode,
            inputs=[audio_mode],
            outputs=[single_mode_group, role_mode_group]
        )
        
        # Audio generation event
        generate_btn.click(
            fn=process_single_audio_generation,
            inputs=[
                text_input,
                audio_mode,
                prompt_text_single,
                prompt_audio_single,
                prompt_text_1,
                prompt_audio_1,
                prompt_text_2,
                prompt_audio_2,
                use_normalize_single
            ],
            outputs=[output_audio, status_info],
            show_progress=True
        )
    
    return demo

# Main function
if __name__ == "__main__":
    demo = create_gradio_interface()
    
    # Launch interface
    demo.launch()
