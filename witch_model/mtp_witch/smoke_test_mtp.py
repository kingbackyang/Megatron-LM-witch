# File: smoke_test_mtp.py
import os
import torch
from torch.utils.data import Dataset

from megatron.training import get_args, print_rank_0, pretrain
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

from pretrain_witch_real_mtp import model_provider


class MockGPTDataset(Dataset):
    def __init__(self, length, seq_len, vocab_size):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return {"text": data}


class MockTokenizer:
    def __init__(self, vocab_size):
        self.eod = vocab_size - 1
        self.pad = vocab_size - 1


def train_valid_test_datasets_provider(num_samples):
    args = get_args()

    # Initialize a lightweight tokenizer for mask generation.
    global tokenizer
    tokenizer = MockTokenizer(args.padded_vocab_size)

    dataset_len = max(32, args.global_batch_size * 4)
    dataset = MockGPTDataset(
        length=dataset_len,
        seq_len=args.seq_length,
        vocab_size=args.padded_vocab_size,
    )
    return dataset, dataset, dataset


def get_batch(data_iterator):
    args = get_args()
    keys = ["text"]
    datatype = torch.int64

    data = None
    if data_iterator is not None:
        try:
            data = next(data_iterator)
        except StopIteration:
            pass

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    if data_b is None:
        return None, None, None, None, None

    tokens = data_b["text"].long().cuda()
    tokens = tokens.transpose(0, 1).contiguous()
    labels = tokens[1:].contiguous()
    tokens = tokens[:-1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=tokenizer.eod,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
    )
    return tokens, labels, loss_mask, attention_mask, position_ids


_MTP_CHECK_DONE = False


def _mtp_smoke_check(model):
    global _MTP_CHECK_DONE
    if _MTP_CHECK_DONE:
        return
    _MTP_CHECK_DONE = True

    args = get_args()
    if args.mtp_num_layers is None:
        raise RuntimeError("MTP smoke test requires --mtp-num-layers to be set.")

    if not getattr(model, "mtp_process", False) or not hasattr(model, "mtp"):
        raise RuntimeError("MTP is not enabled on the model.")

    if len(model.mtp.layers) != args.mtp_num_layers:
        raise RuntimeError(
            f"MTP layer count mismatch: expected {args.mtp_num_layers}, "
            f"got {len(model.mtp.layers)}"
        )

    tracker = MTPLossLoggingHelper.tracker
    if "values" not in tracker:
        raise RuntimeError("MTP loss tracker missing; MTP forward likely not executed.")
    if tracker["values"].shape[0] != args.mtp_num_layers:
        raise RuntimeError(
            f"MTP loss tracker shape mismatch: expected {args.mtp_num_layers}, "
            f"got {tracker['values'].shape[0]}"
        )

    print_rank_0("MTP smoke check passed.")


def forward_step(data_iterator, model):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
    )

    _mtp_smoke_check(model)

    def loss_func(output_tensor):
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output_tensor.float(), labels
        )
        loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.sum()
        return loss, {"lm loss": loss}

    return output_tensor, loss_func


if __name__ == "__main__":
    # Run in distributed mode to exercise MTP end-to-end.
    train_valid_test_datasets_provider.is_distributed = True

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
