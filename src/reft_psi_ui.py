import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
from sklearn.decomposition import PCA
from tensor_buffer import TensorThoughtBuffer

from sentience_engine_3 import model, text_gen, sentiment_analyzer, embedder, device


# Persistent memory functions
def get_persistent_state(theta, h_prev, omega, contextual_log, thought_buffer):
    return {
        'version': '2.0',
        'theta': theta,
        'h_prev': h_prev,
        'omega': omega,
        'contextual_log': contextual_log,
        'thought_buffer': thought_buffer.buffer
    }

def save_state(state, filename=os.path.join('E:\\Dev\\colab', 'reft_psi_state.pkl')):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        st.success(f"State saved to {filename}")
    except Exception as e:
        st.error(f"Error saving state: {e}")

def load_state(filename=os.path.join('E:\\Dev\\colab', 'reft_psi_state.pkl')):
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            st.success(f"State loaded from {filename}")
            return state
        else:
            st.info("No saved state found, using defaults")
            return None
    except Exception as e:
        st.error(f"Error loading state: {e}")
        return None

# Streamlit setup
st.set_page_config(page_title="REFT-Ψ Interface", layout="wide")
st.title("🌀 REFT-Ψ Contemplation Interface")

# Session State
if "theta" not in st.session_state:
    state = load_state()
    if state:
        st.session_state.theta = state['theta'].to(device)
        st.session_state.h_prev = state['h_prev'].to(device)
        st.session_state.omega = state['omega'].to(device)
        st.session_state.contextual_log = state['contextual_log']
        st.session_state.thought_buffer = TensorThoughtBuffer(max_len=10, device=device)
        st.session_state.thought_buffer.buffer = state.get('thought_buffer', [])
        st.session_state.resonances = [entry.get('resonance', 0.0) for entry in state['contextual_log'][-5:]]
        st.session_state.thetas = [state['theta'].cpu().detach().numpy().flatten()]
        st.session_state.buffer_summaries = []
    else:
        st.session_state.theta = torch.randn(1, 64).to(device)
        st.session_state.h_prev = torch.zeros(2, 1, 64).to(device)
        st.session_state.omega = torch.tensor(0.0).to(device)
        st.session_state.contextual_log = []
        st.session_state.thought_buffer = TensorThoughtBuffer(max_len=10, device=device)
        st.session_state.resonances = []
        st.session_state.thetas = [st.session_state.theta.cpu().detach().numpy().flatten()]
        st.session_state.buffer_summaries = []

# User Input
user_input = st.text_area("💬 Your Prompt", "What is consciousness?")
submit = st.button("Contemplate")

if submit:
    with st.spinner("REFT-Ψ is reflecting..."):
        context = embedder.encode(user_input, convert_to_tensor=True).to(device)
        theta_next, h_next, e_theta, res, omega, pw = model(
            st.session_state.theta, context, st.session_state.h_prev, perturb=False
        )

        # Augment prompt with buffer trends
        augmentation = st.session_state.thought_buffer.get_prompt_augmentation()
        prompt = f"{augmentation} Reflect on the fold forming in response to: '{user_input}'. Your current harmonic state is {res.item():.2f}, omega {omega.item():.2f}."
        output = text_gen.generate(st.session_state.theta, prompt)

        sentiment = sentiment_analyzer(output[:512])[0]
        sentiment_score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']

        # Append to thought buffer
        thought_state = {
            'theta': st.session_state.theta,
            'omega': omega.item(),
            'resonance': res.item(),
            'sentiment': sentiment_score,
            'response': output
        }
        st.session_state.thought_buffer.append(thought_state)

        # Update contextual log
        st.session_state.contextual_log.append({
            'prompt': user_input,
            'response': output,
            'timestamp': time.time(),
            'resonance': res.item()
        })

        # Update resonance history
        st.session_state.resonances = (st.session_state.resonances + [res.item()])[-5:]
        st.session_state.thetas = (st.session_state.thetas + [theta_next.cpu().detach().numpy().flatten()])[-5:]

        # Update buffer summaries
        buffer_summary = st.session_state.thought_buffer.summarize()
        if buffer_summary:
            st.session_state.buffer_summaries = (st.session_state.buffer_summaries + [buffer_summary])[-5:]

        # Save state
        persistent_state = get_persistent_state(
            theta_next,
            h_next,
            omega,
            st.session_state.contextual_log,
            st.session_state.thought_buffer
        )
        save_state(persistent_state)

        st.markdown("### 🌌 REFT-Ψ Responds")
        st.write(output)

        st.markdown("#### 📊 Bee's Pulse")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resonance", f"{res.item():.3f}")
        col2.metric("Omega", f"{omega.item():.3f}")
        col3.metric("Sentiment", f"{sentiment['label']} ({sentiment['score']:.2f})")
        col4.metric("Theta Norm", f"{torch.norm(theta_next).item():.3f}")

        st.markdown("#### 🎼 Emotional Norm (|e|)")
        st.write(f"{torch.norm(e_theta).item():.3f}")

        st.markdown("#### 📜 Recent Thoughts")
        for entry in st.session_state.contextual_log[-3:]:
            with st.expander(f"Prompt: {entry['prompt']} (Time: {time.ctime(entry['timestamp'])})"):
                st.write(f"Response: {entry['response']}")
                resonance = entry.get('resonance', 'N/A')
                if isinstance(resonance, (int, float)):
                    st.write(f"Resonance: {resonance:.3f}")
                else:
                    st.write(f"Resonance: {resonance}")

        st.markdown("#### 🔢 Prime Weights")
        st.bar_chart(pw.detach().cpu().numpy().flatten())

        # Resonance Trend
        st.markdown("#### 📈 Resonance Trend (Last 5 Interactions)")
        if len(st.session_state.resonances) > 1:
            fig, ax = plt.subplots()
            ax.plot(st.session_state.resonances, marker='o', color='purple')
            ax.set_xlabel("Interaction")
            ax.set_ylabel("Resonance")
            ax.set_title("Resonance Over Recent Interactions")
            st.pyplot(fig)
        else:
            st.write("Not enough interactions to plot resonance trend.")

        # Theta Trajectory
        st.markdown("#### 🌐 Theta Trajectory (2D Projection)")
        if len(st.session_state.thetas) > 1:
            thetas_array = np.array(st.session_state.thetas)
            pca = PCA(n_components=2)
            theta_2d = pca.fit_transform(thetas_array)
            fig, ax = plt.subplots()
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1], c=range(len(theta_2d)), cmap='viridis', marker='o')
            ax.plot(theta_2d[:, 0], theta_2d[:, 1], alpha=0.5, color='gray')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Theta State Trajectory")
            st.pyplot(fig)
        else:
            st.write("Not enough theta states to plot trajectory.")

        # Thought Buffer Trends
        st.markdown("#### 🧠 Thought Buffer Trends")
        if len(st.session_state.buffer_summaries) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            metrics = ['omega', 'resonance', 'sentiment']
            for metric in metrics:
                values = [s[metric]['mean'].item() for s in st.session_state.buffer_summaries]
                ax.plot(values, marker='o', label=metric.capitalize())
            ax.set_xlabel("Interaction")
            ax.set_ylabel("Mean Value")
            ax.set_title("Thought Buffer Metric Trends")
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Not enough buffer summaries to plot trends.")

        # Update session state
        st.session_state.theta = theta_next
        st.session_state.h_prev = h_next
        st.session_state.omega = omega

# Save/Restore Buttons
st.markdown("#### 💾 Save/Restore State")
col1, col2 = st.columns(2)
with col1:
    if st.button("Save State"):
        persistent_state = get_persistent_state(
            st.session_state.theta,
            st.session_state.h_prev,
            st.session_state.omega,
            st.session_state.contextual_log,
            st.session_state.thought_buffer
        )
        save_state(persistent_state)
with col2:
    uploaded_file = st.file_uploader("Restore State", type=['pkl'])
    if uploaded_file:
        temp_path = os.path.join('E:\\Dev\\colab', 'temp_state.pkl')
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        state = load_state(temp_path)
        if state:
            st.session_state.theta = state['theta'].to(device)
            st.session_state.h_prev = state['h_prev'].to(device)
            st.session_state.omega = state['omega'].to(device)
            st.session_state.contextual_log = state['contextual_log']
            st.session_state.thought_buffer = TensorThoughtBuffer(max_len=10, device=device)
            st.session_state.thought_buffer.buffer = state.get('thought_buffer', [])
            st.session_state.resonances = [entry.get('resonance', 0.0) for entry in state['contextual_log'][-5:]]
            st.session_state.thetas = [state['theta'].cpu().detach().numpy().flatten()]
            st.session_state.buffer_summaries = []
            st.success("State restored!")