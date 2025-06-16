import streamlit as st
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CoNLL-2003 NER labels
LABEL_NAMES = [
    'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 
    'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'
]

# Color mapping for entities
ENTITY_COLORS = {
    'PER': '#FFB6C1',    # Light Pink for Person
    'ORG': '#98FB98',    # Pale Green for Organization  
    'LOC': '#87CEEB',    # Sky Blue for Location
    'MISC': '#DDA0DD',   # Plum for Miscellaneous
}

# Expanded news examples for both input methods
NEWS_EXAMPLES = {
    "Custom Text": "",  # Empty for user input
    "Business News": "Apple Inc. reported strong quarterly earnings. CEO Tim Cook will speak at the conference in San Francisco next week.",
    "Sports News": "Lionel Messi scored two goals for Paris Saint-Germain against Barcelona at Camp Nou stadium in Spain.",
    "Political News": "President Biden met with German Chancellor Scholz in Washington to discuss NATO policies in Europe.",
    "Technology News": "Google announced new AI features for Gmail users. The company's headquarters in Mountain View will host the developer conference.",
    "Entertainment News": "Netflix announced that director Christopher Nolan will produce a new series. The streaming giant plans to film in Los Angeles and London.",
    "Health News": "The World Health Organization released new guidelines for COVID-19 prevention. Dr. Maria Santos from Johns Hopkins University led the research team.",
    "Finance News": "Tesla stock surged after Elon Musk announced new factory plans in Texas. Goldman Sachs analysts upgraded their rating for the electric vehicle manufacturer.",
    "International News": "The European Union imposed new sanctions on Russia following the meeting in Brussels. French President Macron and Italian Prime Minister Draghi supported the decision.",
    "Science News": "NASA's James Webb Space Telescope captured stunning images of distant galaxies. The Hubble Space Telescope has been operational since 1990, revolutionizing our understanding of the universe.",
    "Education News": "Harvard University announced a new scholarship program for international students. MIT and Stanford University are also expanding their financial aid initiatives."
}

@st.cache_resource
def load_model_and_tokenizer(model_path):
    """Load the trained NER model and tokenizer with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None, None

def predict_entities(text, model, tokenizer):
    """Predict named entities in the given text"""
    try:
        if not text.strip():
            return []
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True,
            return_offsets_mapping=True,  # This helps with position mapping
            return_token_type_ids=False
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_ids = predictions.argmax(dim=-1).squeeze().tolist()
            confidence_scores = predictions.max(dim=-1).values.squeeze().tolist()
        
        # Get offset mapping for accurate position tracking
        offset_mapping = inputs["offset_mapping"].squeeze().tolist()
        
        # Convert single values to lists for consistency
        if isinstance(predicted_ids, int):
            predicted_ids = [predicted_ids]
            confidence_scores = [confidence_scores]
        
        # Extract entities
        entities = []
        current_entity = None
        current_start = None
        current_end = None
        current_tokens = []
        current_confidences = []
        
        for i, (pred_id, confidence, (start_offset, end_offset)) in enumerate(
            zip(predicted_ids, confidence_scores, offset_mapping)
        ):
            # Skip special tokens
            if start_offset == 0 and end_offset == 0:
                continue
                
            label = LABEL_NAMES[pred_id] if pred_id < len(LABEL_NAMES) else 'O'
            
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    entity_text = text[current_start:current_end].strip()
                    if entity_text:
                        entities.append({
                            'text': entity_text,
                            'label': current_entity,
                            'confidence': np.mean(current_confidences),
                            'start': current_start,
                            'end': current_end
                        })
                
                # Start new entity
                current_entity = label[2:]  # Remove B- prefix
                current_start = start_offset
                current_end = end_offset
                current_tokens = [text[start_offset:end_offset]]
                current_confidences = [confidence]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue current entity
                current_end = end_offset
                current_tokens.append(text[start_offset:end_offset])
                current_confidences.append(confidence)
                
            else:
                # End current entity
                if current_entity:
                    entity_text = text[current_start:current_end].strip()
                    if entity_text:
                        entities.append({
                            'text': entity_text,
                            'label': current_entity,
                            'confidence': np.mean(current_confidences),
                            'start': current_start,
                            'end': current_end
                        })
                
                current_entity = None
                current_start = None
                current_end = None
                current_tokens = []
                current_confidences = []
        
        # Handle last entity
        if current_entity:
            entity_text = text[current_start:current_end].strip()
            if entity_text:
                entities.append({
                    'text': entity_text,
                    'label': current_entity,
                    'confidence': np.mean(current_confidences),
                    'start': current_start,
                    'end': current_end
                })
        
        return entities
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error(f"Error during prediction: {e}")
        return []

def highlight_entities(text, entities):
    """Create highlighted HTML text with entities"""
    if not entities:
        return text
    
    # Sort entities by start position (reverse order for replacement)
    entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted_text = text
    for entity in entities_sorted:
        start, end = entity['start'], entity['end']
        entity_text = entity['text']
        label = entity['label']
        confidence = entity['confidence']
        
        # Get color for entity type
        color = ENTITY_COLORS.get(label, '#FFFFE0')  # Default to light yellow
        
        # Create highlighted span
        highlighted_span = f"""<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px; border: 1px solid #ccc;">{entity_text} <small style="color: #666;">({label}: {confidence:.2f})</small></span>"""
        
        # Replace in text
        highlighted_text = highlighted_text[:start] + highlighted_span + highlighted_text[end:]
    
    return highlighted_text

def main():
    # Set page config
    st.set_page_config(
        page_title="Named Entity Recognition App",
        page_icon="🏷️",
        layout="wide"
    )
    
    # Title and description
    st.title("🏷️ Named Entity Recognition (NER) App")
    st.markdown("""
    This app uses a fine-tuned DistilBERT model to identify named entities in text.
    The model can recognize **Persons (PER)**, **Organizations (ORG)**, **Locations (LOC)**, and **Miscellaneous (MISC)** entities.
    """)
    
    # Sidebar for model configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./final_ner_model",
        help="Path to your trained NER model directory"
    )
    
    # Load model button
    if st.sidebar.button("🔄 Load Model"):
        st.session_state.model_loaded = False
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.tokenizer = None
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading model... This may take a moment."):
            model, tokenizer = load_model_and_tokenizer(model_path)
            
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check the model path and ensure the model files exist.")
                st.info("Make sure your model directory contains: config.json, pytorch_model.bin, tokenizer.json, and other necessary files.")
                return
    
    # Main interface
    if st.session_state.model_loaded:
        st.header("📝 Enter Text for NER Analysis")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Text Area with Examples", "Quick Examples"],
            horizontal=True
        )
        
        if input_method == "Text Area with Examples":
            # Add selectbox for news examples above text area
            st.subheader("📰 Select a news example or write your own:")
            
            # News category selection
            selected_news = st.selectbox(
                "Choose a news example:",
                list(NEWS_EXAMPLES.keys()),
                help="Select a pre-filled example or choose 'Custom Text' to write your own"
            )
            
            # Get the example text
            example_text = NEWS_EXAMPLES[selected_news]
            
            # Text area with selected example
            user_text = st.text_area(
                "Enter or edit your text here:",
                value=example_text,
                height=150,
                placeholder="Type or paste your text here for named entity recognition...",
                help="You can edit the selected example or write completely new text"
            )
            
            # Show info about selected example
            if selected_news != "Custom Text":
                st.info(f"📄 Selected: **{selected_news}** - You can edit the text above or select a different example.")
        
        else:
            # Original example method (kept for backward compatibility)
            examples = {
                "Business": "Apple Inc. reported strong quarterly earnings. CEO Tim Cook will speak at the conference in San Francisco next week.",
                "Sports": "Lionel Messi scored two goals for Paris Saint-Germain against Barcelona at Camp Nou stadium in Spain.",
                "Politics": "President Biden met with German Chancellor Scholz in Washington to discuss NATO policies in Europe.",
                "Technology": "Google announced new AI features for Gmail users. The company's headquarters in Mountain View will host the developer conference."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            user_text = st.text_area(
                "Example text (you can edit it):",
                value=examples[selected_example],
                height=100
            )
        
        # Analysis button
        if st.button("🔍 Analyze Text", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    # Get predictions
                    entities = predict_entities(
                        user_text,
                        st.session_state.model,
                        st.session_state.tokenizer
                    )
                    
                    if entities:
                        # Display results
                        st.header("📊 Analysis Results")
                        
                        # Highlighted text
                        st.subheader("🖍️ Highlighted Text")
                        highlighted_html = highlight_entities(user_text, entities)
                        st.markdown(highlighted_html, unsafe_allow_html=True)
                        
                        # Entity summary
                        st.subheader("📋 Detected Entities")
                        
                        # Group entities by type
                        entity_groups = {}
                        for entity in entities:
                            label = entity['label']
                            if label not in entity_groups:
                                entity_groups[label] = []
                            entity_groups[label].append(entity)
                        
                        # Display in columns
                        if entity_groups:
                            cols = st.columns(min(len(entity_groups), 4))  # Max 4 columns
                            for i, (label, group_entities) in enumerate(entity_groups.items()):
                                with cols[i % len(cols)]:
                                    st.markdown(f"**{label}** ({len(group_entities)})")
                                    for entity in group_entities:
                                        confidence_bar = "🟢" if entity['confidence'] > 0.9 else "🟡" if entity['confidence'] > 0.7 else "🔴"
                                        st.write(f"{confidence_bar} {entity['text']} ({entity['confidence']:.2f})")
                        
                        # Detailed table
                        st.subheader("📄 Detailed Results")
                        try:
                            import pandas as pd
                            df = pd.DataFrame(entities)
                            df['confidence'] = df['confidence'].round(3)
                            st.dataframe(df, use_container_width=True)
                        except ImportError:
                            # Fallback if pandas not available
                            for i, entity in enumerate(entities, 1):
                                st.write(f"{i}. **{entity['text']}** - {entity['label']} (confidence: {entity['confidence']:.3f})")
                    
                    else:
                        st.info("ℹ️ No named entities detected in the text.")
                        st.write("This could mean:")
                        st.write("- The text doesn't contain recognizable entities")
                        st.write("- The model confidence is very low")
                        st.write("- Try with different text or check your model")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
    **Legend:**
    - 🟢 High confidence (>0.9)  
    - 🟡 Medium confidence (0.7-0.9)  
    - 🔴 Low confidence (<0.7)
    
    **Entity Types:**
    - **PER**: Person names
    - **ORG**: Organizations, companies, institutions
    - **LOC**: Locations, countries, cities
    - **MISC**: Miscellaneous entities
    """)

if __name__ == "__main__":
    main()