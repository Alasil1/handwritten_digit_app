import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- 1. Model Definition (Must match the training script) ---
latent_dim = 100
image_size = 28
num_channels = 1 # Grayscale

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- 2. Load Model ---
@st.cache_resource # Cache the model loading for performance
def load_generator_model(model_path="generator_epoch_50.pth"): # Assuming you saved it as this
    model = Generator(latent_dim, 10, image_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure 'generator.pth' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

generator_model = load_generator_model()

# --- 3. Streamlit App Interface ---
st.title("Handwritten Digit Generation Web App")
st.write("Select a digit to generate 5 unique handwritten images.")

# Digit selection
selected_digit = st.selectbox(
    "Choose a digit (0-9):",
    options=list(range(10))
)

if st.button("Generate Images"):
    st.write(f"Generating 5 images for digit: **{selected_digit}**")

    # Generate images
    num_images_to_generate = 5
    # Create distinct noise vectors for diversity
    noise_vectors = [torch.randn(1, latent_dim) for _ in range(num_images_to_generate)]
    digit_labels = torch.full((num_images_to_generate,), selected_digit, dtype=torch.long)

    generated_images = []
    with torch.no_grad():
        for i in range(num_images_to_generate):
            # Combine the individual noise vector with the label for the specific image
            img = generator_model(noise_vectors[i], digit_labels[i:i+1]) # Pass single label for single noise
            generated_images.append(img.squeeze().numpy()) # Remove batch and channel dims, convert to numpy

    # Display images
    st.subheader("Generated Images:")
    cols = st.columns(5) # Create 5 columns for images

    for i, img_array in enumerate(generated_images):
        # Convert the [-1, 1] normalized image back to [0, 255] for display
        img_display = ((img_array * 0.5 + 0.5) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_display)
        cols[i].image(pil_img, caption=f"Image {i+1}", use_column_width=True)

    st.success("Images generated successfully!")

st.markdown(
    """
    ---
    **Note:** The model was trained from scratch on the MNIST dataset using PyTorch.
    Accuracy is based on visual recognition; some variations are expected as per design.
    """
)