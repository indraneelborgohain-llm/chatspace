"""
server.py - Streamlit server for GPT-OSS text generation
A simple web interface to interact with your trained model
"""
import streamlit as st
import torch
import os
from pathlib import Path

from inference import load_model_and_generate
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
def load_model(checkpoint_path, device):
    """Load and cache the model"""
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
    st.title("🤖 GPT-OSS Text Generator")
    st.markdown("Generate creative text using your trained GPT-OSS model")
    
    # Sidebar for model settings
    st.sidebar.header("⚙️ Model Settings")
    
    # Device selection
    device_options = ["cuda:0" if torch.cuda.is_available() else "cpu", "cpu"]
    device = st.sidebar.selectbox("Device", device_options, index=0)
    
    # Checkpoint selection
    st.sidebar.subheader("Model Checkpoint")
    
    # Find available checkpoints
    checkpoint_dir = "model"
    available_checkpoints = []
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".pt"):
                available_checkpoints.append(os.path.join(checkpoint_dir, file))
    
    if available_checkpoints:
        checkpoint_path = st.sidebar.selectbox(
            "Select checkpoint",
            available_checkpoints,
            index=0 if "gptoss_best.pt" not in str(available_checkpoints) else 
                  [i for i, x in enumerate(available_checkpoints) if "best" in x][0]
        )
    else:
        checkpoint_path = st.sidebar.text_input(
            "Checkpoint path",
            value="model/gptoss_best.pt"
        )
    
    # Load model
    model, model_info = load_model(checkpoint_path, device)
    
    if model is None:
        st.error(f"❌ {model_info}")
        st.stop()
    else:
        # Display model info
        st.sidebar.success("✅ Model loaded successfully!")
        if isinstance(model_info, dict):
            st.sidebar.info(f"""
            **Model Info:**
            - Size: {model_info.get('model_size', 'unknown')}
            - Iteration: {model_info.get('iter', 'unknown')}
            - Val Loss: {model_info.get('val_loss', 'unknown')}
            """)
    
    st.sidebar.markdown("---")
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    
    max_tokens = st.sidebar.slider(
        "Max tokens",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Higher values = more random, lower = more focused"
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=500,
        value=200,
        step=10,
        help="Consider only top k most likely tokens"
    )
    
    # Main content area
    st.markdown("---")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Prompt")
        prompt = st.text_area(
            "Enter your prompt:",
            value="Once upon a time",
            height=150,
            help="Enter the text prompt to start generation"
        )
        
        # Generate button
        generate_button = st.button("🚀 Generate Text", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("✨ Generated Output")
        output_container = st.empty()
    
    # Initialize session state for generated text
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    
    # Generate text when button is clicked
    if generate_button:
        if not prompt.strip():
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating text..."):
                generated = generate_from_model(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device
                )
                st.session_state.generated_text = generated
    
    # Display generated text
    if st.session_state.generated_text:
        with col2:
            output_container.text_area(
                "Generated text:",
                value=st.session_state.generated_text,
                height=300,
                label_visibility="collapsed"
            )
            
            # Copy button
            st.download_button(
                label="📋 Download Text",
                data=st.session_state.generated_text,
                file_name="generated_text.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Examples section
    st.markdown("---")
    st.subheader("💡 Example Prompts")
    
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    with examples_col1:
        if st.button("🏰 Fantasy Story", use_container_width=True):
            st.session_state.example_prompt = "In a kingdom far away, there lived a brave knight who"
    
    with examples_col2:
        if st.button("🚀 Sci-Fi Adventure", use_container_width=True):
            st.session_state.example_prompt = "The spaceship landed on the mysterious planet, and the crew discovered"
    
    with examples_col3:
        if st.button("🎭 Mystery Tale", use_container_width=True):
            st.session_state.example_prompt = "The detective examined the clues carefully and realized that"
    
    # Update prompt if example was clicked
    if "example_prompt" in st.session_state:
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Built with ❤️ using Streamlit | GPT-OSS Model
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
