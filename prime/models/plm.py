import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
# from esm.sdk.forge import ESM3ForgeInferenceClient
# from esm.sdk.api import ESMProtein, LogitsConfig
# from esm.models.esmc import ESMC
import re


def EsmModelInfo(name: str):
    """Get model info by name:
    Args:
        name: str, model name
    Returns:
        dict, model info: dim, layers, model
    """
    return {
        "esm2_t48_15B_UR50D": {
            "dim": 5120,
            "layers": 48,
            "model": "facebook/esm2_t48_15B_UR50D",
        },
        "esm2_t36_3B_UR50D": {
            "dim": 2560,
            "layers": 36,
            "model": "facebook/esm2_t36_3B_UR50D",
        },
        "esm2_t33_650M_UR50D": {
            "dim": 1280,
            "layers": 33,
            "model": "facebook/esm2_t33_650M_UR50D",
        },
        "esm2_t30_150M_UR50D": {
            "dim": 640,
            "layers": 30,
            "model": "facebook/esm2_t30_150M_UR50D",
        },
        "esm2_t12_35M_UR50D": {
            "dim": 480,
            "layers": 12,
            "model": "facebook/esm2_t12_35M_UR50D",
        },
        "esm2_t6_8M_UR50D": {
            "dim": 320,
            "layers": 6,
            "model": "facebook/esm2_t6_8M_UR50D",
        },
        "esm1b_t33_650M_UR50S": {
            "dim": 1280,
            "layers": 33,
            "model": "facebook/esm1b_t33_650M_UR50S",
        },
        "prot_t5_xl_half_uniref50-enc": {
            "dim": 1024,
            "layers": 24,
            "model": "Rostlab/prot_t5_xl_uniref50",
        },
        "prot_t5_xl_bfd": {
            "dim": 1024,
            "layers": 24,
            "model": "Rostlab/prot_t5_xl_bfd",
        },
        "esmc-6b-2024-12": {
            "dim": 2560,
            "layers": -1,
            "model": "esmc-6b-2024-12",
        },
        "esmc_300m": {
            "dim": 768,
            "layers": -1,
            "model": "esmc_300m",
        },
        "esmc_600m": {
            "dim": 1152,
            "layers": -1,
            "model": "esmc_600m",
        },
    }[name]


plm2abbr = {
    'esm2_t48_15B_UR50D': 'ESM2_T48',
    'esm2_t36_3B_UR50D': 'ESM2_T36',
    'esm2_t33_650M_UR50D': 'ESM2_T33',
    'esm2_t30_150M_UR50D': 'ESM2_T30',
    'esm2_t12_35M_UR50D': 'ESM2_T12',
    'esm2_t6_8M_UR50D': 'ESM2_T6',
    'esm1b_t33_650M_UR50S': 'ESM1B_T33',
    'prot_t5_xl_half_uniref50-enc': 'PT_UR',
    'prot_t5_xl_bfd': 'PT_BFD',
}


class EsmEncoder(nn.Module):
    def __init__(self, model_name, dev):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            # auto, balanced_low_0
            model_name,
            device_map="balanced",
            torch_dtype=torch.float16,
            offload_folder="offload",
            offload_state_dict=True,
        )
        if model_name == "facebook/esm2_t48_15B_UR50D":
            self.max_len = 512
        else:
            self.max_len = 960
        self.overlap = 31
        self.model.eval()
        self.model.half()

    def forward(self, _seqs):
        with torch.no_grad() and torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            assert len(_seqs) == 1, "currently only support batch size 1"
            seqs = _seqs[0]
            # left overlappping, right overlappping
            seqs = [
                seqs[max(0, i - self.overlap): (i + self.max_len + self.overlap)]
                for i in range(0, len(seqs), self.max_len)
            ]
            segs = []
            for seq in seqs:
                inputs = self.tokenizer(
                    [seq],
                    return_tensors="pt",
                ).to(self.model.device)
                outputs = (
                    self.model(
                        **inputs).last_hidden_state.squeeze(0).detach().cpu()
                )
                outputs0 = self.model.embeddings(
                    **inputs).squeeze(0).detach().cpu()
                segs.append(torch.stack([outputs0, outputs], dim=-1))
            t = []
            for i in range(len(seqs)):
                if i == 0:
                    t.append(segs[i][1: (1 + self.max_len)])
                elif i == len(seqs) - 1:
                    t.append(segs[i][1 + self.overlap:])
                else:
                    t.append(
                        segs[i][1 + self.overlap: 1 +
                                self.max_len + self.overlap]
                    )
            outputs = torch.cat(t, dim=0)[: len(_seqs[0])]
            assert outputs.shape[0] == len(_seqs[0])
            return outputs


class T5Encoder(nn.Module):
    def __init__(self, name: str, dev) -> None:
        super().__init__()
        self.dev = dev
        if name == "Rostlab/prot_t5_xl_uniref50":
            # Load the tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_half_uniref50-enc",
                do_lower_case=False,
                legacy=False,
            )
            # Load the model
            self.model = T5EncoderModel.from_pretrained(
                "Rostlab/prot_t5_xl_half_uniref50-enc"
            ).to(dev)
        elif name == "Rostlab/prot_t5_xl_bfd":
            # Load the tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_bfd",
                do_lower_case=False,
                legacy=False,
            )
            # Load the model
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd").to(
                dev
            )

        self.max_len = 960  # start_token, end_token occupy 2 positions
        self.overlap = 31
        self.model.eval()
        self.model.half()

    def forward(self, _seqs):
        with torch.no_grad():
            assert len(_seqs) == 1, "currently only support batch size 1"
            seqs = _seqs[0]
            # replace non-amino acids with X
            seqs = re.sub(r"[^A-Z]", "X", seqs)
            # left overlappping, right overlappping
            seqs = [
                seqs[max(0, i - self.overlap)
                         : (i + self.max_len + self.overlap)]
                for i in range(0, len(seqs), self.max_len)
            ]
            input_ids = self.tokenizer.batch_encode_plus(
                [" ".join(list(s)) for s in seqs],
                add_special_tokens=True,
                padding="longest",
            )["input_ids"]
            input_ids = torch.tensor(input_ids).to(self.dev)
            outputs = self.model(input_ids=input_ids)
            outputs0 = self.model.get_input_embeddings()(input_ids)
            outputs = outputs.last_hidden_state
            outputs = torch.stack([outputs0, outputs], dim=-1)
            t = []
            for i in range(len(seqs)):
                if i == 0:
                    t.append(outputs[i, 1: (1 + self.max_len)])
                elif i == len(seqs) - 1:
                    t.append(outputs[i, 1 + self.overlap:])
                else:
                    t.append(
                        outputs[i, 1 + self.overlap: 1 +
                                self.max_len + self.overlap]
                    )

            outputs = torch.cat(t, dim=0)[: len(_seqs[0])]
            assert outputs.shape[0] == len(_seqs[0]), \
                f"outputs shape {outputs.shape} does not match input seqs length {len(_seqs[0])}: {seqs}"
            return outputs


def get_model(name: str, dev):
    "Get model by name"
    if name in [
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
        "esm1b_t33_650M_UR50S",
    ]:
        d = EsmModelInfo(name)
        return EsmEncoder(d["model"], dev)
    elif name in ["prot_t5_xl_half_uniref50-enc", "prot_t5_xl_bfd"]:
        d = EsmModelInfo(name)
        return T5Encoder(d["model"], dev)
    else:
        raise ValueError(f"Unknown model name: {name}")
