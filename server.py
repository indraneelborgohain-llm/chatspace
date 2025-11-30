"""
server.py - Streamlit server for GPT-OSS text generation
A simple web interface to interact with your trained model
"""
import streamlit as st
import torch
import os
from pathlib import Path

from inference import load_model_and_generate, load_gptoss20b_and_generate
from architecture.gptoss import Transformer, ModelConfig
from architecture.tokenizer import get_tokenizer


# Page configuration
st.set_page_config(
    page_title="GPT-OSS Text Generator",
    page_icon="🤖",
    layout="wide"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_sapphire_model(checkpoint_path, device):
    """Load and cache the Sapphire model"""
    try:
        if not os.path.exists(checkpoint_path):
            return None, f"Checkpoint not found: {checkpoint_path}"
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        vocab_size = 201088
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config_dict = checkpoint["config"]
            model_size = config_dict.get("model_size", "toy")
        else:
            model_size = "toy"
        
        # Build config
        from train import build_config
        cfg = build_config(model_size, vocab_size)
        
        # Create and load model
        model = Transformer(cfg, device=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            iter_num = checkpoint.get("iter_num", "unknown")
            val_loss = checkpoint.get("best_val_loss", "unknown")
        else:
            model.load_state_dict(checkpoint)
            iter_num = "unknown"
            val_loss = "unknown"
        
        model.eval()
        
        return model, {"iter": iter_num, "val_loss": val_loss, "model_size": model_size}
    
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def generate_from_model(model, prompt, max_tokens, temperature, top_k, device):
    """Generate text from the loaded model"""
    from inference import generate_text
    
    try:
        with torch.no_grad():
            generated = generate_text(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
        return generated
    except Exception as e:
        return f"Error generating text: {str(e)}"


def main():
    # Title and description
    st.title("💎 Sapphire Text Generator")
    st.markdown("Generate creative text using AI")
    
    # Model selection at the top
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["Sapphire (Custom)", "GPT-OSS 20B"],
            index=0
        )
    
    with col2:
        if model_type == "Sapphire (Custom)":
            # Find available checkpoints
            checkpoint_dir = "model"
            available_checkpoints = []
            if os.path.exists(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    if file.endswith(".pt"):
                        available_checkpoints.append(os.path.join(checkpoint_dir, file))
            
            if available_checkpoints:
                checkpoint_path = st.selectbox(
                    "Checkpoint",
                    available_checkpoints,
                    index=0 if "gptoss_best.pt" not in str(available_checkpoints) else 
                          [i for i, x in enumerate(available_checkpoints) if "best" in x][0]
                )
            else:
                checkpoint_path = st.text_input(
                    "Checkpoint path",
                    value="model/gptoss_best.pt"
                )
        else:
            weights_dir = st.text_input(
                "Weights directory",
                value="architecture/open-gpt-oss/weights"
            )
    
    with col3:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        st.text_input("Device", value=device, disabled=True)
    
    st.markdown("---")
    
    # Main prompt area (larger)
    prompt = st.text_area(
        "Enter your prompt:",
        value="Once upon a time",
        height=120,
        placeholder="Type your prompt here..."
    )
    
    # Generation parameters in expandable section
    with st.expander("⚙️ Advanced Settings"):
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            max_tokens = st.slider(
                "Max tokens",
                min_value=10,
                max_value=500,
                value=200,
                step=10
            )
        
        with param_col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1
            )
        
        with param_col3:
            top_k = st.slider(
                "Top-k",
                min_value=1,
                max_value=500,
                value=200,
                step=10
            )
    
    # Generate button (full width, prominent)
    generate_button = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    # Initialize session state for generated text
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    
    # Generate text when button is clicked
    if generate_button:
        if not prompt.strip():
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating..."):
                try:
                    if model_type == "Sapphire (Custom)":
                        # Load Sapphire model
                        model, model_info = load_sapphire_model(checkpoint_path, device)
                        if model is None:
                            st.error(f"❌ {model_info}")
                            st.stop()
                        
                        generated = generate_from_model(
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            device=device
                        )
                    else:
                        # GPT-OSS 20B
                        generated = load_gptoss20b_and_generate(
                            weights_dir=weights_dir,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            device=device
                        )
                    
                    st.session_state.generated_text = generated
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.stop()
    
    # Display generated text
    if st.session_state.generated_text:
        st.markdown("### ✨ Generated Text")
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; 
            font-family: monospace; white-space: pre-wrap;'>{st.session_state.generated_text}</div>""",
            unsafe_allow_html=True
        )
        
        # Download button
        st.download_button(
            label="📋 Download",
            data=st.session_state.generated_text,
            file_name="generated_text.txt",
            mime="text/plain"
        )
    
    # Examples at the bottom
    st.markdown("---")
    st.markdown("**💡 Try these prompts:**")
    
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    with examples_col1:
        if st.button("🏰 Fantasy", use_container_width=True):
            st.session_state.example_prompt = "In a kingdom far away, there lived a brave knight who"
            st.rerun()
    
    with examples_col2:
        if st.button("🚀 Sci-Fi", use_container_width=True):
            st.session_state.example_prompt = "The spaceship landed on the mysterious planet, and the crew discovered"
            st.rerun()
    
    with examples_col3:
        if st.button("🎭 Mystery", use_container_width=True):
            st.session_state.example_prompt = "The detective examined the clues carefully and realized that"
            st.rerun()
    
    # Update prompt if example was clicked
    if "example_prompt" in st.session_state:
        prompt = st.session_state.example_prompt
        del st.session_state.example_prompt


if __name__ == "__main__":
    main()
