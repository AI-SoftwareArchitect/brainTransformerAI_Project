# main.py - BrainFormer Advanced Implementation
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# CUDA OPTIMIZATION & SETUP
# =============================================================================
# CUDA kontrolü ve optimizasyon
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    print(f"🚀 CUDA kullanılıyor: {torch.cuda.get_device_name()}")
    # CUDA bellek optimizasyonu
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")
    print("⚠️ CPU kullanılıyor")

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class BrainFormerConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    vocab_size: int = 1000
    max_seq_len: int = 512
    memory_size: int = 128
    ff_hidden_dim: int = 1024  # FFN genişliği
    early_exit_threshold: float = 0.9
    dropout: float = 0.3
    temperature: float = 0.8
    device: str = str(device)

# =============================================================================
# ADVANCED MULTIHEAD ATTENTION
# =============================================================================
class DirectedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Context bias for directed attention
        self.context_bias = nn.Parameter(torch.zeros(embed_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better training"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                context_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.num_heads
        head_dim = self.head_dim
        
        # Projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Context-directed attention
        if context_vector is not None:
            Q = Q + self.context_bias * context_vector
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        K = K.view(B, T, H, head_dim).transpose(1, 2)
        V = V.view(B, T, H, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# =============================================================================
# ADVANCED MEMORY SYSTEMS
# =============================================================================
class ShortTermMemory(nn.Module):
    """Enhanced STM with gating mechanism"""
    def __init__(self, embed_dim: int, memory_size: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        
        # Learnable memory bank
        self.memory = nn.Parameter(torch.randn(1, memory_size, embed_dim) * 0.1)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Expand memory for batch
        memory = self.memory.expand(B, -1, -1)
        
        # Gate-controlled memory integration
        gate_scores = self.gate(x)  # (B, T, 1)
        gated_x = x * gate_scores
        
        # Combine with memory
        combined = torch.cat([gated_x, memory], dim=1)
        return combined

class LongTermMemory(nn.Module):
    """Enhanced LTM with attention-based retrieval"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Memory transformation layers
        self.memory_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Circular buffer for long-term storage
        self.register_buffer('ltm_buffer', torch.zeros(1024, embed_dim))
        self.register_buffer('ltm_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x: torch.Tensor, store: bool = False) -> torch.Tensor:
        if store and self.training:
            # Store important representations
            with torch.no_grad():
                # Simple importance scoring (can be improved)
                importance = torch.norm(x, dim=-1, keepdim=True)
                important_items = x[importance.squeeze(-1) > importance.mean()]
                
                if len(important_items) > 0:
                    # Store in circular buffer
                    ptr = self.ltm_ptr.item()
                    batch_size = min(important_items.size(0), 32)  # Batch size limit
                    
                    end_ptr = (ptr + batch_size) % 1024
                    if end_ptr > ptr:
                        self.ltm_buffer[ptr:end_ptr] = important_items[:batch_size]
                    else:
                        self.ltm_buffer[ptr:] = important_items[:1024-ptr]
                        if end_ptr > 0:
                            self.ltm_buffer[:end_ptr] = important_items[1024-ptr:batch_size]
                    
                    self.ltm_ptr[0] = end_ptr
        
        return self.memory_transform(x)

class CombinedAttention(nn.Module):
    """Fusion mechanism for STM and LTM outputs"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Learnable fusion weights
        self.stm_weight = nn.Parameter(torch.ones(1))
        self.ltm_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, stm_output: torch.Tensor, ltm_output: torch.Tensor) -> torch.Tensor:
        # Weighted combination
        weighted_stm = self.stm_weight * stm_output
        weighted_ltm = self.ltm_weight * ltm_output
        
        # Concatenate and fuse
        combined = torch.cat([weighted_stm, weighted_ltm], dim=-1)
        return self.fusion(combined)

# =============================================================================
# EARLY EXIT CLASSIFIER
# =============================================================================
class EarlyExitClassifier(nn.Module):
    """Confidence-based early exit mechanism"""
    def __init__(self, embed_dim: int, vocab_size: int, threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, vocab_size)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        logits = self.classifier(x)  # [B, seq_len, vocab_size]
        confidence = self.confidence_head(x).mean()
        should_exit = confidence.item() > self.threshold
        return logits, should_exit

# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================
class TransformerBlock(nn.Module):
    """Enhanced transformer block with residual connections"""
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.attention = DirectedMultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),  # Better activation than ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

# =============================================================================
# DUAL BRAIN ARCHITECTURE
# =============================================================================
class DualBrainTransformer(nn.Module):
    """Left (Logic) and Right (Creative) brain processing"""
    def __init__(self, config: BrainFormerConfig):
        super().__init__()
        self.config = config
        
        # Left brain: Logical, structured processing
        self.left_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Right brain: Creative, noisy processing
        self.right_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.noise_scale = 0.1
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        left_x = x
        right_x = x
        
        # Process through both hemispheres
        for left_block, right_block in zip(self.left_blocks, self.right_blocks):
            # Left brain: structured processing
            left_x = left_block(left_x, mask)
            
            # Right brain: creative processing with noise injection
            noise = torch.randn_like(right_x) * self.noise_scale
            right_x = right_block(right_x + noise, mask)
        
        return left_x, right_x

# =============================================================================
# MAIN BRAINFORMER MODEL
# =============================================================================
class BrainFormer(nn.Module):
    """Advanced Brain-like Transformer Architecture"""
    def __init__(self, config: BrainFormerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Dual brain architecture
        self.dual_brain = DualBrainTransformer(config)
        
        # Memory systems
        self.stm = ShortTermMemory(config.d_model, config.memory_size)
        self.ltm = LongTermMemory(config.d_model)
        self.combined_attention = CombinedAttention(config.d_model)
        
        # Early exit mechanism
        self.early_exit = EarlyExitClassifier(config.d_model, config.vocab_size, config.early_exit_threshold)
        
        # Output layers
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb

        # Create causal mask
        mask = self._create_causal_mask(seq_len).to(input_ids.device)

        # Dual brain processing
        left_output, right_output = self.dual_brain(x, mask)

        # Memory integration (her token için)
        stm_out = self.stm(left_output)    # [B, seq_len+memory_size, d_model]
        ltm_out = self.ltm(right_output, store=self.training)  # [B, seq_len, d_model]

        # Sadece ilk seq_len token'ı kullan (STM ve LTM'nin ilk seq_len kısmı)
        stm_out = stm_out[:, :seq_len, :]  # [B, seq_len, d_model]
        ltm_out = ltm_out[:, :seq_len, :]  # [B, seq_len, d_model]

        # Combine memory outputs (her token için)
        combined = self.combined_attention(stm_out, ltm_out)  # [B, seq_len, d_model]

        # Early exit check (her token için)
        early_logits, should_exit = self.early_exit(combined)  # [B, seq_len, vocab_size]
        if should_exit and not self.training:
            return early_logits  # [B, seq_len, vocab_size]

        # Final processing
        final_output = self.final_norm(combined)
        logits = self.output_proj(final_output)  # [B, seq_len, vocab_size]
        return logits

# =============================================================================
# ADVANCED SAMPLING
# =============================================================================
def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    """Advanced sampling with temperature, top-k, and top-p (nucleus sampling)"""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, -float('inf'))
        logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
        logits = logits_filtered

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        # İşlemi her örnek için ayrı yap
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Top-p maskesi oluştur
        sorted_indices_to_remove = cumulative_probs > top_p
        # Her satırda ilk token daima tutulmalı
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Maskeyi orijinal indekslere uygula
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float('inf')

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# =============================================================================
# DATA HANDLING (Enhanced)
# =============================================================================
def load_data(file_path: str = "data.txt"):
    """Enhanced data loading with punctuation removal and better English tokenization"""
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("""Your English training data here...""")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove punctuation and tokenize
    text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuation
    sentences = re.split(r'[\n]', text.strip())  # Split by line for sentences
    all_words = []
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        all_words.extend(words)
    
    vocab = sorted(set(all_words))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    # Convert to token sequences
    token_sequences = []
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        if words:
            tokens = [word_to_id[word] for word in words]
            token_sequences.append(tokens)
    
    return token_sequences, vocab, word_to_id, id_to_word

def create_batches(token_sequences, seq_len=16, batch_size=4):
    """Create training batches from token sequences"""
    batches = []
    
    # Create input-target pairs
    pairs = []
    for sequence in token_sequences:
        if len(sequence) > seq_len:
            for i in range(len(sequence) - seq_len):
                input_seq = sequence[i:i+seq_len]
                target_seq = sequence[i+1:i+seq_len+1]
                pairs.append((torch.tensor(input_seq), torch.tensor(target_seq)))
    
    # Group into batches
    for i in range(0, len(pairs), batch_size):
        batch_group = pairs[i:i+batch_size]
        if len(batch_group) == batch_size:  # Only use complete batches
            inputs = torch.stack([b[0] for b in batch_group])
            targets = torch.stack([b[1] for b in batch_group])
            batches.append((inputs, targets))
    
    return batches

# =============================================================================
# ADVANCED TRAINING
# =============================================================================
def train_model(model, batches, epochs=50, lr=1e-4):
    """Enhanced training with advanced optimizer and scheduler"""
    if not batches:
        print("❌ No batches available for training!")
        return
        
    # Advanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_input, batch_target in batches:
            batch_input = batch_input.to(device)   # [B, seq_len]
            batch_target = batch_target.to(device) # [B, seq_len]

            optimizer.zero_grad()

            logits = model(batch_input)  # [B, seq_len, vocab_size]
            # CrossEntropyLoss expects [N, C] and targets [N]
            logits = logits.view(-1, logits.size(-1))      # [(B*seq_len), vocab_size]
            targets = batch_target.view(-1)                # [(B*seq_len)]

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================
def save_model(model, vocab, word_to_id, id_to_word, path="brainformer_advanced.pt"):
    """Save model with enhanced metadata"""
    # config'i dict olarak kaydet
    config_dict = model.config.__dict__ if hasattr(model.config, '__dict__') else dict(model.config)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'config': config_dict,
        'model_class': 'BrainFormer'
    }, path)
    print(f"✅ Gelişmiş model kaydedildi: {path}")

def load_model(path="brainformer_advanced.pt"):
    """Load model with error handling"""
    if not os.path.exists(path):
        return None, None, None, None

    try:
        # PyTorch 2.6+ için weights_only=False ekle
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_dict = checkpoint['config']
        # Dict'ten BrainFormerConfig nesnesi oluştur
        config = BrainFormerConfig(**config_dict)
        config.device = str(device)

        model = BrainFormer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint['vocab'], checkpoint['word_to_id'], checkpoint['id_to_word']
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return None, None, None, None

# =============================================================================
# ADVANCED INFERENCE
# =============================================================================
def generate_text(model, vocab, word_to_id, id_to_word, prompt="hello", max_length=30, temperature=0.8):
    model.eval()
    words = re.findall(r"\b\w+\b", prompt.lower())
    input_ids = [word_to_id[word] for word in words if word in word_to_id]
    if not input_ids:
        return "<No valid prompt tokens in vocab!>"
    generated = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor(generated[-16:]).unsqueeze(0).to(device)
            logits = model(input_tensor)  # [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]
            next_token_id = sample_with_temperature(
                next_token_logits.unsqueeze(0),
                temperature=temperature,
                top_k=50,
                top_p=0.9
            ).item()
            generated.append(next_token_id)
            if len(generated) >= 50:
                break
    generated_words = [id_to_word.get(token_id, "<UNK>") for token_id in generated]
    return " ".join(generated_words)

# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    print("🧠 BrainFormer Advanced - Beyin Benzeri Transformer")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    while True:
        print("\n1 - Train (Gelişmiş Eğitim)")
        print("2 - Run (Gelişmiş Çalıştırma)")
        print("3 - Exit (Çıkış)")
        
        choice = input("\nSeçiminiz (1-3): ").strip()
        
        if choice == "1":
            print("\n🔄 Gelişmiş eğitim başlıyor...")
            
            # Load data
            token_sequences, vocab, word_to_id, id_to_word = load_data()
            print(f"📊 Veri yüklendi - Vocab: {len(vocab)}, Sequences: {len(token_sequences)}")
            
            # Create config with smaller sequence length
            config = BrainFormerConfig(
                vocab_size=len(vocab),
                max_seq_len=16,
                d_model=128,         # Küçültülmüş model boyutu
                num_heads=4,
                num_layers=4,
                ff_hidden_dim=512,   # Küçültülmüş FFN
                dropout=0.1
            )
            
            # Create model
            model = BrainFormer(config).to(device)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"🏗️ Model oluşturuldu - Parametreler: {param_count:,}")
            
            # Create batches with smaller sequence length
            batches = create_batches(token_sequences, seq_len=16, batch_size=8)
            print(f"📦 Batch'ler hazırlandı: {len(batches)}")
            
            if len(batches) == 0:
                print("❌ Yeterli veri yok!")
                continue
            
            # Train
            train_model(model, batches, epochs=30, lr=5e-4)  # Reduced epochs
            
            # Save
            save_model(model, vocab, word_to_id, id_to_word)
            print("✅ Gelişmiş eğitim tamamlandı!")
            
        elif choice == "2":
            print("\n🚀 Gelişmiş model çalıştırılıyor...")
            
            # Load model
            model, vocab, word_to_id, id_to_word = load_model()
            
            if model is None:
                print("❌ Eğitilmiş model bulunamadı! Önce eğitim yapın.")
                continue
            
            print("✅ Gelişmiş model yüklendi!")
            print("🎛️ Gelişmiş sampling aktif (temperature, top-k, top-p)")
            
            while True:
                prompt = input("\nPrompt girin (çıkmak için 'q'): ")
                if prompt.lower() == 'q':
                    break
                
                # Generate with different temperatures
                print(f"\n🌡️ Temperature Karşılaştırması:")
                for temp in [0.3, 0.8, 1.2]:
                    generated_text = generate_text(model, vocab, word_to_id, id_to_word, prompt, temperature=temp)
                    print(f"T={temp}: {generated_text}")
                
        elif choice == "3":
            print("\n👋 Görüşmek üzere!")
            break
            
        else:
            print("❌ Geçersiz seçim! Lütfen 1, 2 veya 3 girin.")

if __name__ == "__main__":
    main()