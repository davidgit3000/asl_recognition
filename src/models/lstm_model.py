"""
LSTM-based model for ASL recognition.
"""
import torch
import torch.nn as nn


class ASLLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for ASL recognition from landmark sequences.
    
    Input: [batch, time, landmarks, channels] = [B, T, 75, 4]
    Output: [batch, num_classes]
    """
    
    def __init__(
        self,
        num_classes: int = 45,
        input_dim: int = 300,  # 75 landmarks * 4 channels
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes
            input_dim: Input feature dimension (75 * 4 = 300)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, time, landmarks, channels] = [B, T, 75, 4]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len, num_landmarks, num_channels = x.shape
        
        # Flatten landmarks and channels: [B, T, 75*4]
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM: output = [B, T, hidden_dim * 2]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state (or mean pooling)
        # For bidirectional: concatenate forward and backward last states
        if self.bidirectional:
            # h_n shape: [num_layers * 2, B, hidden_dim]
            # Take last layer's forward and backward hidden states
            forward_hidden = h_n[-2, :, :]  # [B, hidden_dim]
            backward_hidden = h_n[-1, :, :]  # [B, hidden_dim]
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, hidden_dim*2]
        else:
            last_hidden = h_n[-1, :, :]  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(last_hidden)  # [B, num_classes]
        
        return logits


class ASLLSTMWithAttention(nn.Module):
    """
    LSTM with temporal attention for ASL recognition.
    """
    
    def __init__(
        self,
        num_classes: int = 45,
        input_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, time, landmarks, channels] = [B, T, 75, 4]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len, num_landmarks, num_channels = x.shape
        
        # Flatten: [B, T, 300]
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM: [B, T, hidden_dim*2]
        lstm_out, _ = self.lstm(x)
        
        # Attention weights: [B, T, 1]
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum: [B, hidden_dim*2]
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classify
        logits = self.classifier(context)
        
        return logits


def create_model(model_type: str = "lstm", num_classes: int = 45, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: 'lstm' or 'lstm_attention'
        num_classes: Number of output classes
        **kwargs: Additional model arguments
    
    Returns:
        model: PyTorch model
    """
    if model_type == "lstm":
        return ASLLSTMModel(num_classes=num_classes, **kwargs)
    elif model_type == "lstm_attention":
        return ASLLSTMWithAttention(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
