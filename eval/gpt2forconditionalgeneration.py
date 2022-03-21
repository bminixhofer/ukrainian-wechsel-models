from json import decoder
import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.file_utils import is_torch_fx_proxy


class GPT2ForConditionalGeneration(GPT2LMHeadModel):
    # from T5ForConditialGeneration
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(
                input_ids.shape[:-1] + (1,), decoder_start_token_id
            )
            shifted_input_ids = torch.cat(
                [shifted_input_ids, input_ids[..., :-1]], dim=-1
            )
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs
    ):
        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if attention_mask is None:
            attention_mask = (
                torch.ones_like(input_ids) if input_ids is not None else None
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = (
                torch.ones_like(decoder_input_ids)
                if decoder_input_ids is not None
                else None
            )

        input_ids = torch.cat(
            ([input_ids] if input_ids is not None else [])
            + ([decoder_input_ids] if decoder_input_ids is not None else []),
            1,
        )
        attention_mask = torch.cat(
            ([attention_mask] if input_ids is not None else [])
            + ([decoder_attention_mask] if decoder_input_ids is not None else []),
            1,
        )

        output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        lm_logits = output.logits[
            :, -(decoder_input_ids.shape[1] if decoder_input_ids is not None else 1) :
        ]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
        )
