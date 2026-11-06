import pytest
import torch
from tasks.autoresearch.worker import validate_model_contract


class ToyAttention(torch.nn.Module):
    def __init__(self, leak=False, loss_scale=1):
        super().__init__()
        self.leak, self.loss_scale = leak, loss_scale

    def forward(self, idx, targets=None, reduction='mean'):
        values = torch.nn.functional.one_hot(idx, 16).float().unsqueeze(1)
        n = idx.shape[1]
        row, col = torch.arange(n)[:, None], torch.arange(n)[None, :]
        allowed = (col <= row) & (row-col <= 3)
        # Reproduce the observed candidate's exact Boolean-mask inversion.
        mask = ~allowed if self.leak else allowed
        logits = torch.nn.functional.scaled_dot_product_attention(values, values, values, attn_mask=mask).squeeze(1)
        if targets is None:
            return logits
        return self.loss_scale * torch.nn.functional.cross_entropy(logits.reshape(-1,16),targets.reshape(-1),reduction=reduction)


@pytest.mark.parametrize('leak,scale,error', [(False,1,None), (True,1,'Non-causal'), (False,.5,'unmodified')])
def test_causal_and_loss_contract(leak, scale, error):
    x=torch.arange(16).reshape(1,16)
    y=(x+1)%16
    if error:
        with pytest.raises(ValueError,match=error):
            validate_model_contract(ToyAttention(leak,scale),x,y,16)
    else:
        result=validate_model_contract(ToyAttention(),x,y,16)
        assert result['passed'] and result['prefix_max_difference']==0


class ChunkLeak(torch.nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, idx, targets=None, reduction='mean'):
        values = torch.nn.functional.one_hot(idx, 16).float()
        batch, length, vocab = values.shape
        chunks = values.reshape(batch, -1, self.chunk_size, vocab)
        logits = chunks.mean(dim=2, keepdim=True).expand_as(chunks).reshape_as(values)
        if targets is None:
            return logits
        return torch.nn.functional.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1), reduction=reduction)


@pytest.mark.parametrize('chunk_size', [2, 4, 8])
def test_chunk_leak_that_passes_original_midpoint_is_rejected(chunk_size):
    x = torch.arange(16).reshape(1, 16)
    changed = x.clone(); changed[:, 8:] = (changed[:, 8:]+1)%16
    model = ChunkLeak(chunk_size)
    assert torch.equal(model(x)[:, :8], model(changed)[:, :8])
    with pytest.raises(ValueError, match='Non-causal'):
        validate_model_contract(model, x, (x+1)%16, 16)


def test_causal_probe_covers_chunk_interiors_without_altering_inputs():
    x = torch.arange(16).reshape(1, 16); original = x.clone()
    result = validate_model_contract(ToyAttention(), x, (x+1)%16, 16)
    assert [row['cut'] for row in result['prefix_checks']] == [1, 7, 8, 15]
    assert torch.equal(x, original)
