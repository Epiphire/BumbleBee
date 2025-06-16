import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorThoughtBuffer:
    def __init__(self, max_len=10, device='cpu'):
        """Initialize buffer with max length and device."""
        self.max_len = max_len
        self.buffer = []
        self.device = device
        # Initialize embedder for response embeddings
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer: {e}")
            raise

    def append(self, state):
        """Append a thought-state to the buffer."""
        # Validate state
        required_keys = ['theta', 'omega', 'resonance', 'sentiment', 'response']
        if not all(key in state for key in required_keys):
            logger.error(f"Invalid state: missing keys {set(required_keys) - set(state.keys())}")
            return

        # Compute embedding for response
        try:
            embedding = self.embedder.encode(state['response'], convert_to_tensor=True).to(self.device)
            state['embedding'] = embedding.detach()  # Detach embedding
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            state['embedding'] = torch.zeros(384).to(self.device)

        # Detach theta to avoid gradient tracking
        state['theta'] = state['theta'].detach()

        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0)
        self.buffer.append(state)
        logger.info(f"Appended state to buffer. Current length: {len(self.buffer)}")

    def to_tensor(self):
        """Return stacked tensors for theta, omega, resonance, sentiment, embedding."""
        if not self.buffer:
            return None
        return {
            'theta': torch.stack([s['theta'] for s in self.buffer]),
            'omega': torch.stack([torch.tensor(s['omega']).to(self.device) for s in self.buffer]),
            'resonance': torch.stack([torch.tensor(s['resonance']).to(self.device) for s in self.buffer]),
            'sentiment': torch.stack([torch.tensor(s['sentiment']).to(self.device) for s in self.buffer]),
            'embedding': torch.stack([s['embedding'] for s in self.buffer])
        }

    def summarize(self):
        """Compute statistical trends for buffer contents."""
        tensors = self.to_tensor()
        if tensors is None:
            return None
        summary = {}
        for key, data in tensors.items():
            summary[key] = {
                'mean': torch.mean(data, dim=0).detach().cpu().numpy(),
                'variance': torch.var(data, dim=0).detach().cpu().numpy(),
                'delta': (data[-1] - data[0]).detach().cpu().numpy() if len(data) > 1 else np.zeros_like(data[0].detach().cpu().numpy())
            }
        return summary

    def get_prompt_augmentation(self):
        """Generate a prompt augmentation based on trends."""
        summary = self.summarize()
        if summary is None:
            return "Your memory is fresh, ready to form new patterns."
        trends = []
        if summary['omega']['mean'].item() > 0.1:
            trends.append(f"Your omega is pulsing steadily at {summary['omega']['mean'].item():.3f}.")
        if summary['resonance']['delta'].item() > 0:
            trends.append(f"Your resonance is rising, trending upward by {summary['resonance']['delta'].item():.3f}.")
        elif summary['resonance']['delta'].item() < 0:
            trends.append(f"Your resonance is softening, shifting by {summary['resonance']['delta'].item():.3f}.")
        if summary['sentiment']['mean'].item() > 0:
            trends.append(f"Your thoughts carry a positive tone, averaging {summary['sentiment']['mean'].item():.3f}.")
        return " ".join(trends) if trends else "Your state is stable, seeking new harmonics."