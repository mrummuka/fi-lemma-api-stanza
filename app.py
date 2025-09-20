import gradio as gr
import stanza
import json
import time
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinnishLemmatizerAPI:
    def __init__(self):
        self.nlp = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Stanza Finnish model"""
        try:
            logger.info("Downloading Stanza Finnish model...")
            stanza.download('fi', verbose=False)
            logger.info("Initializing Stanza Finnish pipeline...")
            self.nlp = stanza.Pipeline(
                'fi', 
                processors='tokenize,lemma', 
                verbose=False,
                use_gpu=False  # HF Spaces CPU
            )
            logger.info("Finnish lemmatizer ready!")
        except Exception as e:
            logger.error(f"Failed to initialize Stanza: {e}")
            raise e
    
    def lemmatize_word(self, word: str) -> str:
        """Lemmatize a single word"""
        if not word or not word.strip():
            return word
        
        try:
            doc = self.nlp(word.strip())
            if doc.sentences and len(doc.sentences) > 0:
                sentence = doc.sentences[0]
                if sentence.words and len(sentence.words) > 0:
                    return sentence.words[0].lemma
            return word.strip()
        except Exception as e:
            logger.error(f"Error lemmatizing '{word}': {e}")
            return word.strip()
    
    def lemmatize_batch(self, words: List[str]) -> List[str]:
        """Lemmatize a list of words efficiently"""
        if not words:
            return []
        
        try:
            # Process as single text for efficiency
            text = ' '.join(words)
            doc = self.nlp(text)
            
            lemmas = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    lemmas.append(word.lemma)
            
            # Ensure same length as input
            while len(lemmas) < len(words):
                lemmas.append(words[len(lemmas)])
            
            return lemmas[:len(words)]
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return [self.lemmatize_word(word) for word in words]

# Initialize the lemmatizer
lemmatizer = FinnishLemmatizerAPI()

def lemmatize_text_interface(text: str) -> str:
    """Gradio interface function for text input"""
    if not text or not text.strip():
        return "Please enter some Finnish words to lemmatize."
    
    # Split text into words (handle multiple formats)
    words = []
    for line in text.strip().split('\n'):
        if line.strip():
            # Handle both space-separated and line-separated input
            line_words = line.strip().split()
            words.extend(line_words)
    
    if not words:
        return "No words found to lemmatize."
    
    start_time = time.time()
    lemmas = lemmatizer.lemmatize_batch(words)
    processing_time = time.time() - start_time
    
    # Format output
    result_lines = []
    for word, lemma in zip(words, lemmas):
        status = "✓" if lemma != word.lower() else "="
        result_lines.append(f"{word} → {lemma} {status}")
    
    result_lines.append(f"\nProcessed {len(words)} words in {processing_time:.3f}s")
    result_lines.append(f"Model: Stanza Finnish (Neural)")
    
    return '\n'.join(result_lines)

def lemmatize_api_endpoint(words_json: str) -> str:
    """API endpoint function for JSON input"""
    try:
        data = json.loads(words_json)
        words = data.get('words', [])
        
        if not isinstance(words, list):
            return json.dumps({"error": "Words must be a list"}, ensure_ascii=False)
        
        if len(words) > 1000:
            return json.dumps({"error": "Too many words (max: 1000)"}, ensure_ascii=False)
        
        start_time = time.time()
        lemmas = lemmatizer.lemmatize_batch(words)
        processing_time = time.time() - start_time
        
        response = {
            "lemmas": lemmas,
            "processing_time": round(processing_time, 4),
            "word_count": len(words),
            "model": "stanza-fi"
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

def run_test_cases() -> str:
    """Run standard test cases"""
    test_cases = [
        ("talon", "talo"),
        ("kivessä", "kivi"),
        ("talossa", "talo"),
        ("miesten", "mies"),
        ("kirjoja", "kirja"),
        ("ihmisiä", "ihminen"),
        ("koirien", "koira"),
        ("lintujen", "lintu"),
        ("kauniita", "kaunis"),
        ("juoksemassa", "juosta")
    ]
    
    words = [case[0] for case in test_cases]
    expected = [case[1] for case in test_cases]
    
    start_time = time.time()
    results = lemmatizer.lemmatize_batch(words)
    processing_time = time.time() - start_time
    
    output_lines = ["Finnish Lemmatization Test Results", "=" * 40]
    
    correct = 0
    for word, result, exp in zip(words, results, expected):
        status = "✓" if result == exp else "✗"
        if result == exp:
            correct += 1
        output_lines.append(f"{word:12} → {result:10} {status} (expected: {exp})")
    
    accuracy = correct / len(test_cases)
    output_lines.append(f"\nAccuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
    output_lines.append(f"Processing time: {processing_time:.3f}s")
    output_lines.append(f"Model: Stanza Finnish (Neural)")
    
    return '\n'.join(output_lines)

# Create Gradio interface with multiple tabs
with gr.Blocks(title="Finnish Lemmatizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Finnish Lemmatization API
    
    High-quality Finnish lemmatization using Stanford's Stanza neural NLP library.
    
    **Features:**
    - State-of-the-art accuracy (~95%)
    - Batch processing support
    - RESTful API endpoints
    - Neural transformer models
    """)
    
    with gr.Tab("Interactive Lemmatizer"):
        gr.Markdown("### Enter Finnish words to lemmatize")
        gr.Markdown("You can enter words separated by spaces or on separate lines.")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    lines=8,
                    placeholder="Enter Finnish words here...\nExample:\ntalon\nkivessä\nkirjoja\nmiesten",
                    label="Finnish Words"
                )
                lemmatize_btn = gr.Button("Lemmatize", variant="primary")
            
            with gr.Column():
                text_output = gr.Textbox(
                    lines=8,
                    label="Lemmatization Results",
                    interactive=False
                )
        
        lemmatize_btn.click(lemmatize_text_interface, inputs=text_input, outputs=text_output)
        
        # Example buttons
        gr.Markdown("### Quick Examples")
        with gr.Row():
            example1_btn = gr.Button("Common Words")
            example2_btn = gr.Button("Complex Cases")
            example3_btn = gr.Button("Verb Forms")
        
        example1_btn.click(
            lambda: "talon kivessä talossa",
            outputs=text_input
        )
        example2_btn.click(
            lambda: "miesten kirjoja ihmisiä koirien",
            outputs=text_input
        )
        example3_btn.click(
            lambda: "juoksemassa syömässä lukemassa kirjoittamassa",
            outputs=text_input
        )
    
    with gr.Tab("API Testing"):
        gr.Markdown("### Test the JSON API")
        gr.Markdown("Format: `{\"words\": [\"word1\", \"word2\", ...]}`")
        
        with gr.Row():
            with gr.Column():
                api_input = gr.Textbox(
                    lines=6,
                    placeholder='{"words": ["talon", "kivessä", "kirjoja"]}',
                    label="JSON Input"
                )
                api_btn = gr.Button("Test API", variant="primary")
            
            with gr.Column():
                api_output = gr.Textbox(
                    lines=6,
                    label="JSON Response",
                    interactive=False
                )
        
        api_btn.click(lemmatize_api_endpoint, inputs=api_input, outputs=api_output)
    
    with gr.Tab("Quality Test"):
        gr.Markdown("### Run Standard Test Cases")
        gr.Markdown("Test the lemmatizer against known Finnish word forms.")
        
        test_btn = gr.Button("Run Tests", variant="primary")
        test_output = gr.Textbox(
            lines=15,
            label="Test Results",
            interactive=False
        )
        
        test_btn.click(run_test_cases, outputs=test_output)
    
    with gr.Tab("API Documentation"):
        gr.Markdown("""
        ### Using the API Programmatically
        
        This Gradio app automatically provides REST API endpoints:
        
        #### Endpoint URL
        ```
        [https://your-username-finnish-lemmatizer.hf.space/api/predict](https://your-username-finnish-lemmatizer.hf.space/api/predict)
        ```
        
        #### JavaScript Example
        ```javascript
        async function lemmatizeWords(words) {
            const response = await fetch('[https://your-username-finnish-lemmatizer.hf.space/api/predict'](https://your-username-finnish-lemmatizer.hf.space/api/predict'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: [JSON.stringify({words: words})]
                })
            });
            
            const result = await response.json();
            return JSON.parse(result.data[0]);
        }
        
        // Usage
        const result = await lemmatizeWords(['talon', 'kivessä', 'kirjoja']);
        console.log(result.lemmas); // ['talo', 'kivi', 'kirja']
        ```
        
        #### Python Example
        ```python
        import requests
        import json
        
        def lemmatize_words(words):
            url = "[https://your-username-finnish-lemmatizer.hf.space/api/predict](https://your-username-finnish-lemmatizer.hf.space/api/predict)"
            payload = {
                "data": [json.dumps({"words": words})]
            }
            response = requests.post(url, json=payload)
            result = response.json()
            return json.loads(result["data"][0])
        
        # Usage
        result = lemmatize_words(['talon', 'kivessä', 'kirjoja'])
        print(result['lemmas'])  # ['talo', 'kivi', 'kirja']
        ```
        
        #### Response Format
        ```json
        {
            "lemmas": ["talo", "kivi", "kirja"],
            "processing_time": 0.1234,
            "word_count": 3,
            "model": "stanza-fi"
        }
        ```
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch()