# Section 0: Standard library imports
import sys
import re
import os
from pathlib import Path
from typing import Optional, Dict

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Section 1: Configuration
# Where to find prompts and where to save norms
prompts_dir = ROOT_DIR / "LLM prompts"
norms_dir = ROOT_DIR / "utils" / "norms"

# Default OpenAI configuration
OPENAI_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7


# Section 2: Loading Prompts
def list_prompts():
    """List all available prompt files in LLM prompts folder."""
    if not prompts_dir.exists():
        print("No 'LLM prompts' folder found.")
        return []
    
    # Find all .txt files
    prompt_files = sorted([f.name for f in prompts_dir.glob("*.txt")])
    
    if not prompt_files:
        print("No .txt prompt files found in LLM prompts folder.")
        return []
    
    print(f"\n{'='*60}")
    print("AVAILABLE PROMPTS:")
    print(f"{'='*60}")
    for i, prompt_file in enumerate(prompt_files, 1):
        print(f"  {i}. {prompt_file}")
    print(f"{'='*60}\n")
    
    return prompt_files


def load_prompt(prompt_file: str) -> str:
    """
    Load prompt text from LLM prompts folder.
    
    Args:
        prompt_file: Filename (e.g., "Commons Open.txt")
        
    Returns:
        The full prompt text as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_path = prompts_dir / prompt_file
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# Section 3: Code Extraction (Simplified - just returns the full response)
# Alerative: Extract code section of the output prompt, would ratehr just imporve prompt for now
def extract_class_name(response: str) -> Optional[str]:
    """
    Try to extract the class name from the response (for logging).
    
    This is optional - just helps us name the file better.
    
    Args:
        response: Full text response from LLM
        
    Returns:
        Class name if found, or None
    """
    match = re.search(r'class\s+(\w+).*?Norm', response)
    return match.group(1) if match else None


# Section 4: API Calling Functions
# These functions actually talk to the LLM APIs
def call_openai_api(prompt: str, model: str, api_key: str, temperature: float = 0.7) -> str:
    """
    Call OpenAI API (for GPT-4, GPT-5, etc.).
    
    How it works:
    1. Import the openai library (you need: pip install openai)
    2. Create a "client" object with your API key (like opening a connection)
    3. Call the API method with your prompt
    4. Get back the generated text
    
    Args:
        prompt: The text prompt to send to the LLM
        model: Which model to use (e.g., "gpt-4-turbo")
        api_key: Your secret API key (from environment variable)
        temperature: How creative (0.0 = deterministic, 1.0 = creative, default 0.7)
        
    Returns:
        The generated text response from the LLM
        
    Raises:
        ImportError: If openai package not installed
        Exception: If API call fails (wrong key, network error, etc.)
    """
    try:
        import openai
        
        # Create client - this is like "connecting" to OpenAI's servers
        # The api_key proves you're authorized to use their service
        client = openai.OpenAI(api_key=api_key)
        
        # Make the API call
        # chat.completions.create() sends your prompt to GPT and gets response
        response = client.chat.completions.create(
            model=model,  # Which GPT model to use
            messages=[{"role": "user", "content": prompt}],  # Your prompt
            temperature=temperature,  # Use configurable temperature
        )
        
        # Extract the text from the response object
        # Response structure: response.choices[0].message.content
        return response.choices[0].message.content
        
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    except Exception as e:
        raise Exception(f"OpenAI API error: {e}")


# Section 5: Main Workflow
# This function orchestrates everything: loads prompt, calls API, saves response
def generate_norm(
    prompt_file: Optional[str] = None,
    norm_name: Optional[str] = None,
    temperature: Optional[float] = None,
    model: str = OPENAI_MODEL,
) -> Dict:
    """
    Generate a norm using LLM API - main entry point.
    
    Flow:
    1. Get API key from environment variable
    2. Load prompt (default to first available prompt if not provided)
    3. Call OpenAI ChatGPT (gpt-4o by default)
    4. Save response to norm file
    
    Args:
        prompt_file: Optional prompt filename (defaults to first prompt in the folder)
        norm_name: Optional custom name (defaults to gpt iteration counter)
        temperature: Optional override (defaults to DEFAULT_TEMPERATURE)
        model: OpenAI model identifier (defaults to gpt-4o)
        
    Returns:
        Dictionary with:
        - norm_file: Path to saved norm file
        - class_name: Name of norm class (if found)
        - llm_response: Full response from LLM
    """
    # Step 1: Validate LLM name
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Set OPENAI_API_KEY environment variable.\n"
            "Example: export OPENAI_API_KEY='your-api-key-here'"
        )
    
    final_temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    print(f"\n{'='*60}")
    print("GENERATING NORM WITH CHATGPT")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Temperature: {final_temperature}")
    print(f"{'='*60}\n")
    
    # Step 3: Load prompt
    print("1. Loading prompt...")
    if prompt_file:
        prompt = load_prompt(prompt_file)
    else:
        prompts = list_prompts()
        if not prompts:
            raise ValueError("No prompts available to use.")
        prompt_file = prompts[0]
        prompt = load_prompt(prompt_file)
    
    print(f"   ✓ Loaded prompt: {prompt_file} ({len(prompt)} characters)\n")
    
    # Step 4: Call ChatGPT
    print("2. Calling ChatGPT API...")
    response = call_openai_api(
        prompt=prompt,
        model=model,
        api_key=api_key,
        temperature=final_temperature,
    )
    print(f"   ✓ Received response ({len(response)} characters)\n")
    
    # Step 5: Try to extract class name (optional, for logging)
    class_name = extract_class_name(response)
    if class_name:
        print(f"   Found class: {class_name}\n")
    
    # Step 6: Determine filename for the norm
    if norm_name is None:
        existing_norms = list(norms_dir.glob("gpt*.py"))
        iteration = len(existing_norms) + 1
        norm_name = f"gpt_iteration{iteration}"
    
    norm_file = norms_dir / f"{norm_name}.py"
    
    # Step 7: Save the response to file
    print(f"3. Saving norm to {norm_file.name}...")
    with open(norm_file, 'w', encoding='utf-8') as f:
        f.write(response)
    print(f"   ✓ Saved\n")
    
    # Return results
    print(f"{'='*60}")
    print(f"✓ NORM GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Norm file: {norm_file}")
    if class_name:
        print(f"Class name: {class_name}")
    print(f"Next: Review the file, then run 'python scripts/experiment.py' to test")
    print(f"{'='*60}\n")
    
    return {
        "norm_file": str(norm_file),
        "class_name": class_name,
        "llm_response": response,
    }


# Section 6: Command-line interface
# Allows running the script from terminal with arguments, or fully interactive
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate norm using OpenAI ChatGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage (first prompt, default temperature)
  python scripts/llm_norm_generator.py
  
  # With specific prompt
  python scripts/llm_norm_generator.py --prompt "Commons Open.txt"

  # With custom temperature
  python scripts/llm_norm_generator.py --temperature 0.5

  # With custom model override
  python scripts/llm_norm_generator.py --model "gpt-4.1-mini"
        """
    )
    
    parser.add_argument(
        "--prompt", 
        help="Prompt filename in LLM prompts folder (defaults to first prompt found)"
    )
    parser.add_argument(
        "--name", 
        help="Custom norm name (defaults to gpt_iteration{N})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature (0.0-1.0, defaults to 0.7)"
    )
    parser.add_argument(
        "--model",
        default=OPENAI_MODEL,
        help=f"OpenAI model identifier (defaults to {OPENAI_MODEL})"
    )
    
    args = parser.parse_args()
    
    result = generate_norm(
        prompt_file=args.prompt,
        norm_name=args.name,
        temperature=args.temperature,
        model=args.model,
    )
    
    print(f"Generated norm: {result['norm_file']}")
