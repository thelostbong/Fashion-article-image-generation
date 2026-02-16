import re, os
os.environ["HF_HOME"] = "/home/nayeemuddin.mohammed/chache"
os.environ["HF_DATASETS_CACHE"] = "/home/nayeemuddin.mohammed/chache"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, CLIPModel
import json
import pandas as pd
from transformers import pipeline
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
import statistics
import random


class ProductPromptGenerator:
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct", device=None):
        """
        Initialize the prompt generator with a LLM and CLIP model.
        Args:
            model_name: The model to use (default: phi-3-mini)
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map=self.device
        )
        
        # Initialize CLIP model for scoring (using larger model for better performance)
        print("Loading CLIP model for scoring...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
    def generate_response(self, prompt, max_length=1024, temperature=0.0):
        """Generate a response from the model with the given prompt."""
        # Set seed for text generation reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=False if temperature == 0.0 else True,
                top_p=0.95 if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]


         #"""
        #debug LLM 
        print("PROMPT INPUT START_________________________________________________________")
        print(prompt)
        print("PROMPT INPUT END_________________________________________________________")
        print("-" * 80)
        print("LLM OUTPUT START_________________________________________________________")
        print(response)
        print("LLM OUTPUT END_________________________________________________________")
        #"""


        
        return response.strip()
    
    def extract_product_attributes(self, product_description):
        """Extract key product attributes using the LLM."""
        extraction_prompt = f"""<|system|>
You are an AI assistant that extracts structured product information from descriptions.
<|user|>
Extract the following elements from this product description:
1. Product type (what exactly is it?)
2. Key visual features (2-3 most distinctive visual elements)
3. Color and material details
4. Any unique design aspects

Product description:
{product_description}

Return the extracted elements in JSON format with these exact keys:
- product_type (simple text string)
- key_visual_features (simple comma-separated text string)
- color_material_details (simple comma-separated text string)
- design_aspects (simple comma-separated text string)

Do NOT include lists, arrays, or nested objects in your JSON. Use only simple text strings.
<|assistant|>"""

        response = self.generate_response(extraction_prompt)

        # Debug output
        print("LLM RESPONSE RAW:")
        print(response)
        print("-" * 80)

        # Try to isolate the first valid JSON block manually
        start_idx = response.find('{')
        end_idx = response.find('}', start_idx)
    
        if start_idx == -1 or end_idx == -1:
            print("JSON block not found — falling back to manual parser.")
            return self._create_structured_data_from_text(response)

        # Try to expand to the full JSON block in case it spans multiple lines
        brace_stack = []
        end_idx = None
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_stack.append('{')
            elif response[i] == '}':
                brace_stack.pop()
                if not brace_stack:
                    end_idx = i
                    break

        if end_idx is None:
            print("JSON block appears incomplete — falling back to manual parser.")
            return self._create_structured_data_from_text(response)

        json_str = response[start_idx:end_idx+1]

        print("EXTRACTED JSON BLOCK:")
        print(json_str)
        print("-" * 80)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("JSON decode failed — falling back to manual parser.")
            return self._create_structured_data_from_text(response)


    def _create_structured_data_from_text(self, text):
        """Fallback method to extract structured data from non-JSON text."""
        data = {
            "product_type": "",
            "key_visual_features": "",
            "color_material_details": "",
            "design_aspects": ""
        }
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if "product type" in line.lower() or "product:" in line.lower():
                data["product_type"] = re.sub(r'^.*?:', '', line).strip()
            elif "visual" in line.lower() or "feature" in line.lower():
                data["key_visual_features"] = re.sub(r'^.*?:', '', line).strip()
            elif "color" in line.lower() or "material" in line.lower():
                data["color_material_details"] = re.sub(r'^.*?:', '', line).strip()
            elif "design" in line.lower() or "aspect" in line.lower():
                data["design_aspects"] = re.sub(r'^.*?:', '', line).strip()
        return data
    
    def clean_text_for_clip(self, text):
        """Clean and optimize text for CLIP scoring."""
        # Remove empty parts and clean formatting
        text = re.sub(r',\s*,', ',', text)  # Remove double commas
        text = re.sub(r',\s*$', '', text)   # Remove trailing comma
        text = re.sub(r'^\s*,', '', text)   # Remove leading comma
        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
        
        # Remove very short or empty descriptors
        parts = [part.strip() for part in text.split(',') if part.strip() and len(part.strip()) > 2]
        return ', '.join(parts) if parts else text.strip()
    
    def calculate_clip_score(self, image, text):
        """Calculate CLIP score between image and text."""
        try:
            # Clean the text for better CLIP scoring
            cleaned_text = self.clean_text_for_clip(text)
            
            # Process inputs
            inputs = self.clip_processor(text=[cleaned_text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get features
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                # Convert to similarity score (0-100 scale)
                clip_score = logits_per_image.squeeze().cpu().item()
                
            return float(clip_score)
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
            return 0.0
    
    def generate_product_image_prompt(self, product_description):
        """Generate a complete image generation prompt from a product description."""
        # Set seed before translation for consistency
        torch.manual_seed(42)
        
        # Extract product attributes
        product_description = self.translator(product_description)[0]['translation_text']            
        extracted_data = self.extract_product_attributes(product_description)
        
        # Format each element properly
        product_type = extracted_data.get('product_type', 'product')
        
        # Handle key_visual_features - could be string, list, or dict
        key_features = extracted_data.get('key_visual_features', 'distinctive features')
        if isinstance(key_features, list):
            key_features = ", ".join(key_features)
        elif isinstance(key_features, dict):
            key_features = ", ".join([f"{k}: {v}" for k, v in key_features.items()])
        
        # Handle color_material_details - could be string, list, or dict
        color_material = extracted_data.get('color_material_details', '')
        print(color_material)
        if isinstance(color_material, dict):
            color_material = ", ".join([f"{v}" for k, v in color_material.items()])
        elif isinstance(color_material, list):
            color_material = ", ".join(color_material)
            
        # Handle design_aspects - could be string, list, or dict
        design = extracted_data.get('design_aspects', '')
        if isinstance(design, list):
            design = ", ".join(design)
        elif isinstance(design, dict):
            design = ", ".join([f"{v}" for k, v in design.items()])
        
        # Build the enhanced prompt for CLIP scoring (more descriptive but still concise)
        simple_prompt = f"Product image of a {product_type}"
        # simple_prompt = f"{product_type} with {key_features}, {color_material} colors, {design} design"
        simple_prompt = re.sub(r'\s+', ' ', simple_prompt)
        simple_prompt = re.sub(r'\s,', ',', simple_prompt)
        simple_prompt = re.sub(r',,', ',', simple_prompt)
        
        # Build the detailed final prompt for image generation
        final_prompt = f"""Professional product photography of a {product_type}, 
featuring {key_features}, {color_material}, {design}, placed on a clean white background, laid flat,
studio lighting, 8k, 85mm lens f/2.8, no humans, no mannequins, no zoom, no tilt, no rotation, perfectly centered"""
        
        # Clean up any redundant spaces or line breaks
        final_prompt = re.sub(r'\s+', ' ', final_prompt)
        final_prompt = re.sub(r'\s,', ',', final_prompt)
        final_prompt = re.sub(r',,', ',', final_prompt)
        
        return final_prompt.strip(), simple_prompt.strip()


if __name__ == "__main__":  
    # Set all random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_path = 'output'
    model_name = "microsoft/phi-3-mini-4k-instruct"
    generator = ProductPromptGenerator(model_name=model_name, device='cuda:0')
    g = torch.Generator(device='cuda:0')

    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16, device_map="balanced")
    pipe.load_lora_weights("aihpi/flux-fashion-lora")
    df = pd.read_excel('article_descriptions_3.xlsx', usecols=[0, 1, 2])
    
    # Initialize results storage
    clip_scores = []
    
    print("Starting image generation and CLIP scoring...")
    print("="*100)
    
    # Set generator seed once for all generations
    g.manual_seed(SEED)
    
    for idx in range(len(df)):
        print(f"Processing sample {idx + 1}/{len(df)}")
        
        # Get original description
        description = df.iloc[idx, :].tolist()
        original_description = '\n'.join([str(item) for item in description if pd.notna(item)])
        
        # Generate prompts (no need to reseed here as we want consistency within run)
        detailed_prompt, simple_prompt = generator.generate_product_image_prompt(original_description)
        print(f"Generated prompt: {detailed_prompt}")
        print(f"CLIP prompt: {simple_prompt}")
        
        # Generate image using detailed prompt with seeded generator
        image = pipe(detailed_prompt, generator=g).images[0]
        
        # Save image
        os.makedirs(save_path, exist_ok=True)
        image_path = f"{save_path}/{idx}.png"
        image.save(image_path)
        
        # Calculate CLIP score using simple prompt
        clip_score = generator.calculate_clip_score(image, simple_prompt)
        clip_scores.append(clip_score)
        
        print(f"CLIP Score: {clip_score:.4f}")
        
        # Save individual JSON for this sample
        result_entry = {
            "sample_id": idx,
            "original_description": original_description,
            "generation_prompt": detailed_prompt,
            "clip_prompt": simple_prompt,
            "clip_score": clip_score,
            "image_path": image_path
        }
        
        json_path = f"{save_path}/{idx}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_entry, f, indent=2, ensure_ascii=False)
        
        print('-'*100)
    
    # Calculate and save statistics
    total_samples = len(clip_scores)
    mean_score = statistics.mean(clip_scores)
    median_score = statistics.median(clip_scores)
    std_dev = statistics.stdev(clip_scores) if len(clip_scores) > 1 else 0.0
    min_score = min(clip_scores)
    max_score = max(clip_scores)
    
    # Simple statistics output
    stats_text = f"""CLIP Score Statistics:
Total Samples: {total_samples}
Mean: {mean_score:.4f}
Median: {median_score:.4f}
Std Dev: {std_dev:.4f}
Min: {min_score:.4f}
Max: {max_score:.4f}
"""
    
    # Print to console
    print("\n" + "="*30)
    print("CLIP SCORE SUMMARY")
    print("="*30)
    print(stats_text)
    
    # Save to text file
    with open(f"{save_path}/clip_statistics.txt", "w", encoding="utf-8") as f:
        f.write(stats_text)
    
    print(f"Individual JSON files saved in: newrun1/")
    print(f"Statistics saved to: clip_statistics.txt")