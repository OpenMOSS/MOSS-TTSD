import os
import torch
import torchaudio
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader
import openai
from generation_utils import load_model, process_batch
import argparse

# =============== Configuration Section ===============
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_32k_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8

# English audio examples
EN_PROMPT_AUDIO_SPEAKER1 = "examples/m1.wav"
EN_PROMPT_TEXT_SPEAKER1 = "How much do you know about her?"
EN_PROMPT_AUDIO_SPEAKER2 = "examples/m2.wav"  
EN_PROMPT_TEXT_SPEAKER2 = "Well, we know this much about her. You've been with her constantly since the first day you met her. And we followed you while you went dining, dancing, and sailing. And last night, I happened to be there when you were having dinner with her at Le Petit Tableau."

# Chinese audio examples
ZH_PROMPT_AUDIO_SPEAKER1 = "examples/pod_f_enhanced.wav"
ZH_PROMPT_TEXT_SPEAKER1 = "但是因为我上学的时候学的是金融学专业嘛，所以确实可能就看起来这个领域跨度还是比较大的。那如果说回我当时的情况的话呢，当时在毕业季的话，大家肯定都会争取拿到更多的offer嘛。"
ZH_PROMPT_AUDIO_SPEAKER2 = "examples/pod_m_enhanced.wav"
ZH_PROMPT_TEXT_SPEAKER2 = "92年回到中国之后跟复旦去建了精算学院，现在跟很多学院学校也有在合作，就是这是最早在国内去做精算的。现在精算也是有中精，就中国精算师体系，还有北美精算师，这个是提的比较多的，也可能是业界含金量最高的吧。"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============== Text Parsing Functions ===============

def extract_text_from_pdf(file_path):
    """Extract text content from PDF file
    
    Args:
        file_path (str): PDF file path
        
    Returns:
        str: Extracted text content, returns None if failed
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF content extraction failed: {str(e)}")
        return None


def extract_web_content(url):
    """Extract title and main content from web URL
    
    Args:
        url (str): Web URL address
        
    Returns:
        tuple: (title, content) - title and main content, returns None if failed
    """
    try:
        # Send request and get web content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'  # Ensure correct encoding
        response.raise_for_status()
        
        print(f"HTTP status: {response.status_code}")
        print(f"Response content length: {len(response.text)}")

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title - try multiple approaches
        title = ""
        
        # Try h1 tags first
        h1_tags = soup.find_all('h1')
        if h1_tags:
            for h1 in h1_tags:
                if h1.text.strip():
                    title = h1.text.strip()
                    break
        
        # Try title tag if h1 not found
        if not title and soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        
        # Try meta property="og:title"
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title['content'].strip()

        print(f"Extracted title: {title}")

        # Simply extract all text from the page
        # Remove script, style, and other non-content elements
        for unwanted in soup.find_all(['script', 'style', 'noscript']):
            unwanted.decompose()
            
        # Get all text content
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Clean the text
        if text_content:
            # Remove extra blank lines
            cleaned_text = re.sub(r'\n{3,}', '\n\n', text_content)
            # Remove extra spaces
            cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
            # Remove very short lines and common noise
            lines = cleaned_text.split('\n')
            filtered_lines = []
            for line in lines:
                line = line.strip()
                # Filter out very short lines and common non-content patterns
                if (len(line) > 3 and 
                    'browser does not support' not in line.lower() and
                    not re.match(r'^[0-9\s\-\/\.]+$', line)):  # Filter date-only lines
                    filtered_lines.append(line)
            
            cleaned_text = '\n'.join(filtered_lines)
            
            print(f"Final content length: {len(cleaned_text)} characters")
            print(f"Content preview: {cleaned_text[:300]}...")
            
            return title, cleaned_text
        else:
            print("No content extracted")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Parse error: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_text_from_txt(file_path):
    """Extract text content from TXT file
    
    Args:
        file_path (str): TXT file path
        
    Returns:
        str: Extracted text content, returns None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            print(f"TXT file reading failed (tried GBK encoding): {str(e)}")
            return None
    except Exception as e:
        print(f"TXT file reading failed: {str(e)}")
        return None


def parse_input_content(input_path):
    """Parse input content, supports URL, PDF or TXT files
    
    Args:
        input_path (str): URL address, PDF file path or TXT file path
        
    Returns:
        str: Parsed text content
    """
    print(f"Parsing input: {input_path}")
    
    # Check if it's a URL
    if input_path.startswith(('http://', 'https://')):
        print("URL detected, extracting web content...")
        result = extract_web_content(input_path)
        if result:
            title, content = result
            print(f"Web title: {title}")
            print(f"Content length: {len(content)} characters")
            return f"{title}\n\n{content}" if title else content
        else:
            print("Web content extraction failed")
            return None
    
    # Check if it's a PDF file
    elif input_path.lower().endswith('.pdf'):
        print("PDF file detected, extracting content...")
        content = extract_text_from_pdf(input_path)
        if content:
            print(f"PDF content length: {len(content)} characters")
            return content
        else:
            print("PDF content extraction failed")
            return None
    
    # Check if it's a TXT file
    elif input_path.lower().endswith('.txt'):
        print("TXT file detected, reading content...")
        content = extract_text_from_txt(input_path)
        if content:
            print(f"TXT file content length: {len(content)} characters")
            return content
        else:
            print("TXT file reading failed")
            return None
    
    else:
        print(f"Unsupported input format: {input_path}")
        return None


# =============== Dialogue Script Generation Function ===============

def generate_podcast_script(content, language='zh'):
    """Call large model to generate podcast dialogue script"""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "YOUR_API_BASE_URL"),
    )

    if language == 'zh':
        role_play = "两位中文播客主持人"
        instruction = f"""你是一位专业的中文播客文字脚本撰稿人。现在请你根据提供的有关最新AI及大模型相关进展的原始资料，生成一段模拟{role_play}之间的自然对话脚本。该脚本应符合以下具体要求：
    一、语言风格
    - 使用较为自然、随意、轻松的日常中文表达；
    - 优先采用简单易懂的词汇，避免书面用语，将书面表达转换为符合口语表达的形式，但不改变专业词汇的内容；
    - 可适度加入网络流行语、俗语、俚语，增强真实感；
    - 符合{role_play}对话的感觉。

    二、句式结构
    - 使用松散、自然的句式，允许存在口语特征如重复、停顿、语气词等；
    - 鼓励使用叠词（如"特别特别"、"慢慢来"）和填充词（如"这个"、"其实"、"然后"、"就是"、"呃"等）；
    - 可适度插入模糊表达、略带情绪的语调，增强亲和力。

    三、对话结构
    - 两个说话人交替发言，并使用[S1]和[S2]标记两位说话人轮次，[S1]和[S2]中间不加入换行；
    - 每当一方讲话时，另一方可以适当插入自然、简短的反馈或承接语（如"嗯。""对。""是的。""确实。""原来是这样。"等），展现倾听状态；
    - 对话应有开头引入、核心讨论与自然结尾，语气上有节奏起伏，避免平铺直叙；
    - 总长度控制在10分钟以内的语音朗读时长（不超过1500字），禁止超时。
    - **特别强调听话方的积极反馈：当一位说话人正在讲述或解释某个观点时，另一位说话人应频繁地插入简短的承接或反馈词语（例如：「嗯。」、「是。」、「对。」、「哦。」、「是的。」、「哦，原来是这样。」、「明白。」、「没错。」、「有道理。」、「确实」），以表明其正在积极倾听、理解和互动。这些反馈应自然地穿插在说话者语句的间歇或段落转换处，而不是生硬地打断。例如：[S2]我本人其实不太相信星座诶，[S1]嗯。[S2]在一开始的时候，我就跟大部分不相信星座的一样，觉得，呃，你总能把人就分成十二种，[S1]是的。[S2]然后呢就它讲的就是对的。这种反馈要尽可能多，不要吝啬。**

    四、标点与格式
    - 仅使用中文标点：逗号、句号、问号；
    - 禁止使用叹号。禁止使用省略号（'...'）、括号、引号（包括''""'"）或波折号等特殊符号；
    - 所有数字转换为中文表达，如"1000000"修改为"一百万"；
    - 请根据上下文，智慧地判断数字的读音，所有带数字的英文缩写要意译，如"a2b"输出为"a到b"、"gpt-4o"输出为"GPT四O"、"3:4"输出为"3比4"，"2021"如果表达年份，应当转换为"二零二一"，但如果表示数字，应当转换为"两千零二十一"。请保证不要简单转换为中文数字，而是根据上下文，将其翻译成合适的中文。

    五、内容要求
    - 所有内容都基于原始资料改写，不得照搬其书面表达，原始资料中所有的内容都要完整的提到，不能丢失或者省略信息；
    - 可加入适当的背景说明、吐槽、对比、联想、提问等方式，增强对话的节奏和趣味性；
    - 确保信息密度较高，引用需确保上下文完整，确保听众能理解；
    - 在对话内不要输出"我是谁"，"我是S1"，"我是S2"等相关内容；
    - 如有专业术语则需要提供解释的解释，如果涉及抽象技术点，可以使用比喻类比等方式解释，避免听起来晦涩难懂。

    ## 原始资料
    {content}

    请根据以上要求和提供的原始资料，将其转化为符合以上所有要求的播客对话脚本。一定要用[S1]和[S2]标记两位说话人，绝对不能使用任何其它符号标记说话人。
    注意：直接输出结果，不要包含任何额外信息。
    """
    else: # Default to English
        role_play = "two English podcast hosts"
        instruction = f"""You are a professional English podcast scriptwriter. Based on the provided source material about the latest developments in AI and large models, generate a natural conversational script simulating a dialogue between {role_play}. The script should meet the following specific requirements:
    I. Language Style
    - Use natural, casual, and relaxed everyday English expressions.
    - Prioritize simple and easy-to-understand vocabulary, avoiding formal language. Convert written expressions into spoken language forms without changing the content of professional terms.
    - Appropriately include internet slang, colloquialisms, and idioms to enhance authenticity.
    - The dialogue should feel like a conversation between {role_play}.

    II. Sentence Structure
    - Use loose and natural sentence structures, allowing for spoken features like repetition, pauses, and filler words.
    - Encourage the use of repetition (e.g., "very, very," "take it slow") and filler words (e.g., "like," "actually," "so," "you know," "uh," etc.).
    - Appropriately insert vague expressions and slightly emotional tones to enhance approachability.

    III. Dialogue Structure
    - The two speakers should take turns speaking, marked with [S1] and [S2] for each turn. Do not add a newline between [S1] and [S2].
    - When one person is speaking, the other can appropriately insert short, natural feedback or connecting phrases (e.g., "Yeah.", "Right.", "Exactly.", "I see.", "Okay.") to show they are listening.
    - The conversation should have an introduction, a core discussion, and a natural conclusion, with a rhythmic and varied tone, avoiding a flat narrative.
    - The total length should be controlled to within a 10-minute reading time (no more than 1500 words). Do not exceed the time limit.
    - **Emphasize active feedback from the listener: When one speaker is explaining a point, the other should frequently interject with short connecting or feedback words (e.g., "Mhm.", "Yeah.", "Right.", "Oh.", "I see.", "Okay.", "Got it.", "Makes sense.", "Totally.") to show active listening, understanding, and engagement. This feedback should be naturally interspersed at pauses or transitions in the speaker's sentences, not as abrupt interruptions. For example: [S2] I'm not a big believer in horoscopes, actually. [S1] Mhm. [S2] At first, like most people who don't believe in them, I thought, uh, you can't just divide people into twelve types, [S1] Right. [S2] and then what it says is just correct. Use this kind of feedback as much as possible; don't be stingy.**

    IV. Punctuation and Formatting
    - Use only standard English punctuation: commas, periods, question marks.
    - Do not use exclamation marks. Do not use special symbols like ellipses ('...'), parentheses, quotation marks (including ''""'"), or dashes.
    - Spell out numbers as words, e.g., "1,000,000" as "one million".
    - Intelligently determine how to pronounce numbers based on context. Spell out abbreviations with numbers, e.g., "a2b" as "a to b", "gpt-4o" as "GPT four O", "3:4" as "three to four". If "2021" is a year, it should be "twenty twenty-one", but if it's a number, it should be "two thousand twenty-one". Ensure you translate it appropriately based on context, not just as a simple conversion.

    V. Content Requirements
    - All content must be rewritten based on the source material; do not copy its written expressions. All information from the source material must be mentioned completely, without omissions.
    - You can add appropriate background explanations, roasts, comparisons, associations, and questions to enhance the rhythm and fun of the dialogue.
    - Ensure high information density and that citations have complete context so the audience can understand.
    - Do not output things like "I am S1" or "I am S2" within the dialogue.
    - If there are technical terms, provide explanations. For abstract technical points, use analogies or metaphors to make them less obscure.

    ## Source Material
    {content}

    Please convert the provided source material into a podcast dialogue script that meets all the above requirements. Be sure to mark the two speakers with [S1] and [S2], and absolutely do not use any other symbols to mark the speakers.
    Note: Output the result directly without any extra information.
    """

    try:
        print("Calling large model to generate dialogue script...")
        print(f"Input content length: {len(content)} characters")
        
        completion = client.chat.completions.create(
            model="gemini-2.5-pro",
            # model="gemini-2.5-flash-preview-04-17",
            messages=[
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        )
        
        raw_result = completion.choices[0].message.content
        
        # Remove all newlines
        processed_result = raw_result.replace('\n', '').replace('\r', '')
        
        print("=" * 50)
        print("Large model generated dialogue script (original version):")
        print("=" * 50)
        print(raw_result)
        print("=" * 50)
        print("Large model generated dialogue script (after removing newlines):")
        print("=" * 50)
        print(processed_result)
        print("=" * 50)
        print(f"Original script length: {len(raw_result)} characters")
        print(f"Processed script length: {len(processed_result)} characters")
        
        return processed_result
        
    except Exception as e:
        print(f"Large model call failed: {str(e)}")
        # If large model call fails, return a sample script for testing
        sample_script = f"[S1]Today we're going to talk about an interesting topic.[S2]Hmm, what topic?[S1]It's about the content we just saw, {content[:100]}...[S2]Oh, sounds very interesting.[S1]Yes, let me give you a detailed introduction."
        print("Using sample script for testing")
        return sample_script


# =============== Main Function ===============

def process_input_to_audio(input_path: str, output_dir: str = "examples", language: str = 'zh'):
    """Complete processing pipeline: from input to audio output
    
    Args:
        input_path (str): Input path (URL, PDF or TXT file)
        output_dir (str): Output directory
        language (str): Language for the podcast script ('en' or 'zh')
    """
    
    # Select prompts based on language
    if language == 'zh':
        prompt_audio_speaker1 = ZH_PROMPT_AUDIO_SPEAKER1
        prompt_text_speaker1 = ZH_PROMPT_TEXT_SPEAKER1
        prompt_audio_speaker2 = ZH_PROMPT_AUDIO_SPEAKER2
        prompt_text_speaker2 = ZH_PROMPT_TEXT_SPEAKER2
    else:  # Default to English
        prompt_audio_speaker1 = EN_PROMPT_AUDIO_SPEAKER1
        prompt_text_speaker1 = EN_PROMPT_TEXT_SPEAKER1
        prompt_audio_speaker2 = EN_PROMPT_AUDIO_SPEAKER2
        prompt_text_speaker2 = EN_PROMPT_TEXT_SPEAKER2
    
    print(f"Using {language} prompts:")
    print(f"Speaker 1: {prompt_audio_speaker1}")
    print(f"Speaker 2: {prompt_audio_speaker2}")
    
    # 1. Parse input content
    print("Step 1: Parse input content")
    content = parse_input_content(input_path)
    if not content:
        print("Unable to parse input content, program terminated")
        return
    
    print(f"Content parsed successfully, content preview: {content[:200]}...")
    
    # 2. Use large model to generate dialogue script
    print("\nStep 2: Generate dialogue script")
    script = generate_podcast_script(content, language=language)
    if not script:
        print("Dialogue script generation failed, program terminated")
        return

    # 3. Load TTS model
    print("\nStep 3: Load TTS model")
    tokenizer, model, spt = load_model(MODEL_PATH, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
    spt = spt.to(device)
    model = model.to(device)
    print("TTS model loading completed")
    
    # 4. Prepare TTS input data with language-specific prompts
    print("\nStep 4: Prepare TTS input data")
    items = [{
        "text": script,
        "base_path": "",
        "prompt_audio_speaker1": prompt_audio_speaker1,
        "prompt_text_speaker1": prompt_text_speaker1,
        "prompt_audio_speaker2": prompt_audio_speaker2,
        "prompt_text_speaker2": prompt_text_speaker2
    }]
    
    # 5. Set random seed
    # import accelerate
    # accelerate.utils.set_seed(42)
    
    # 6. Generate audio
    print("\nStep 5: Generate audio")
    actual_texts_data, audio_results = process_batch(
        batch_items=items,
        tokenizer=tokenizer,
        model=model,
        spt=spt,
        device=device,
        system_prompt=SYSTEM_PROMPT,
        start_idx=0,
        use_normalize=True
    )
    
    # 7. Save audio files
    print("\nStep 6: Save audio files")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, audio_result in enumerate(audio_results):
        if audio_result is not None:
            output_path = os.path.join(output_dir, f"generated_podcast_{idx}.wav")
            torchaudio.save(output_path, audio_result["audio_data"], audio_result["sample_rate"])
            print(f"Audio saved to: {output_path}")
        else:
            print(f"Audio generation failed: sample {idx}")
    
    print("\nProcessing completed!")


# =============== Usage Examples ===============

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Generate podcast audio: supports URL, PDF or TXT file input")
    parser.add_argument("input_path", help="Input path: URL address, PDF file path or TXT file path")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("-l", "--language", default="zh", choices=['en', 'zh'], help="Language of the podcast script (en or zh, default: zh)")
    
    args = parser.parse_args()
    
    # Use command line arguments
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output}")
    print(f"Script language: {args.language}")
    
    process_input_to_audio(args.input_path, args.output, args.language)