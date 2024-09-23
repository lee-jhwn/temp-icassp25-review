
# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# adopted from https://github.com/openspeech-team/openspeech

import random
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor


# from openspeech.modules import (
#     # AdditiveAttention,
#     # DotProductAttention,
#     # Linear,
#     # LocationAwareAttention,
#     # MultiHeadAttention,
#     # View,
# )

class Linear(nn.Module):
    r"""
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    r"""Wrapper class of torch.view() for Sequential module."""

    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs):
        if self.contiguous:
            inputs = inputs.contiguous()
        return inputs.view(*self.shape)

class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        dim (int): The dimension of model (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        self.key_proj = Linear(dim, self.d_head * num_heads)
        self.value_proj = Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = DotProductAttention(dim, scale=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        return context, attn

class LocationAwareAttention(nn.Module):
    r"""
    Applies a location-aware attention mechanism on the output features from the decoders.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        dim (int): dimension of model
        attn_dim (int): dimension of attention
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoders.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoders outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.

    Reference:
        Jan Chorowski et al.: Attention-Based Models for Speech Recognition.
        https://arxiv.org/abs/1506.07503
    """

    def __init__(self, dim: int = 1024, attn_dim: int = 1024, smoothing: bool = False) -> None:
        super(LocationAwareAttention, self).__init__()
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_proj = Linear(dim, attn_dim, bias=False)
        self.value_proj = Linear(dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.fc = Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_alignment_energy: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length)

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(dim=1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)

        alignmment_energy = self.fc(
            torch.tanh(self.query_proj(query) + self.value_proj(value) + last_alignment_energy + self.bias)
        ).squeeze(dim=-1)

        if self.smoothing:
            alignmment_energy = torch.sigmoid(alignmment_energy)
            alignmment_energy = torch.div(alignmment_energy, alignmment_energy.sum(dim=-1).unsqueeze(dim=-1))

        else:
            alignmment_energy = F.softmax(alignmment_energy, dim=-1)

        context = torch.bmm(alignmment_energy.unsqueeze(dim=1), value)

        return context, alignmment_energy

class DotProductAttention(nn.Module):
    r"""
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoders outputs.
    """

    def __init__(self, dim: int, scale: bool = True) -> None:
        super(DotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(query.size()) == 3:
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        else:
            score = torch.matmul(query, key.transpose(2, 3)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)

        if len(query.size()) == 3:
            context = torch.bmm(attn, value)
        else:
            context = torch.matmul(attn, value)

        return context, attn

class AdditiveAttention(nn.Module):
    r"""
    Applies a additive attention (bahdanau) mechanism on the output features from the decoders.
    Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.

    Args:
        dim (int): dimension of model

    Inputs: query, key, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the alignment from the encoders outputs.
    """

    def __init__(self, dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = Linear(dim, dim, bias=False)
        self.key_proj = Linear(dim, dim, bias=False)
        self.score_proj = Linear(dim, 1)
        self.bias = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)

        context += query

        return context, attn

class OpenspeechDecoder(nn.Module):
    r"""Interface of OpenSpeech decoder."""

    def __init__(self):
        super(OpenspeechDecoder, self).__init__()

    def count_parameters(self) -> int:
        r"""Count parameters of decoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        r"""Update dropout probability of decoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class LSTMAttentionDecoder(OpenspeechDecoder):
    r"""
    Converts higher level features (from encoders) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the decoders hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 2)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        attn_mechanism (str, optional): type of attention mechanism (default: multi-head)
        num_heads (int, optional): number of attention heads. (default: 4)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_state_dim): tensor with containing the outputs of the encoders.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: logits
        * logits (torch.FloatTensor) : log probabilities of model's prediction
    """
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        num_classes: int,
        max_length: int = 150,
        hidden_state_dim: int = 1024,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        attn_mechanism: str = "multi-head",
        num_heads: int = 4,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout_p: float = 0.3,
    ) -> None:
        super(LSTMAttentionDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.attn_mechanism = attn_mechanism.lower()
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )

        if self.attn_mechanism == "loc":
            self.attention = LocationAwareAttention(hidden_state_dim, attn_dim=hidden_state_dim, smoothing=False)
        elif self.attn_mechanism == "multi-head":
            self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        elif self.attn_mechanism == "additive":
            self.attention = AdditiveAttention(hidden_state_dim)
        elif self.attn_mechanism == "dot":
            self.attention = DotProductAttention(dim=hidden_state_dim)
        elif self.attn_mechanism == "scaled-dot":
            self.attention = DotProductAttention(dim=hidden_state_dim, scale=True)
        else:
            raise ValueError("Unsupported attention: %s".format(attn_mechanism))

        self.fc = nn.Sequential(
            Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
        self,
        input_var: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        encoder_outputs: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)

        if self.attn_mechanism == "loc":
            context, attn = self.attention(outputs, encoder_outputs, attn)
        else:
            context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        encoder_output_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths: The length of encoders outputs. ``(batch)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()
        hidden_states, attn = None, None

        targets, batch_size, max_length = self.validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            # targets = targets[targets].view(batch_size, -1)
            # print('input var',input_var.size())
            if self.attn_mechanism == "loc" or self.attn_mechanism == "additive":
                for di in range(targets.size(1)):
                    input_var = targets[:, di].unsqueeze(1)
                    step_outputs, hidden_states, attn = self.forward_step(
                        input_var=input_var,
                        hidden_states=hidden_states,
                        encoder_outputs=encoder_outputs,
                        attn=attn,
                    )
                    logits.append(step_outputs)

            else:
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=targets,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                    attn=attn,
                )

                for di in range(step_outputs.size(1)):
                    step_output = step_outputs[:, di, :]
                    logits.append(step_output)

        else:
            input_var = targets[:, 0].unsqueeze(1)
            # print(max_length)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=input_var,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                    attn=attn,
                )
                logits.append(step_outputs)
                input_var = logits[-1].topk(1)[1]
                # print(torch.stack(logits,dim=1).size())


        logits = torch.stack(logits, dim=1)

        return logits

    def validate_args(
        self,
        targets: Optional[Any] = None,
        encoder_outputs: torch.Tensor = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, int, int]:
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:  # inference
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                targets = targets.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1  # minus the start of sequence symbol

        return targets, batch_size, max_length
    
    def next_topk(self, encoder_outputs, encoder_output_lengths, targets):
        logits = list()
        hidden_states, attn = None, None

        targets, batch_size, max_length = self.validate_args(targets, encoder_outputs, 0)
        targets = targets[targets != self.eos_id].view(batch_size, -1)

        for di in range(targets.size(1)):
            input_var = targets[:, di].unsqueeze(1)
            step_outputs, hidden_states, attn = self.forward_step(
                input_var=input_var,
                hidden_states=hidden_states,
                encoder_outputs=encoder_outputs,
                attn=attn,
            )
            logits.append(step_outputs)
        
        logits = torch.stack(logits, dim=1)

        # print(logits.size())

        return logits



class OpenspeechBeamSearchBase(nn.Module):
    """
    Openspeech's beam-search base class. Implement the methods required for beamsearch.
    You have to implement `forward` method.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, decoder, beam_size: int):
        super(OpenspeechBeamSearchBase, self).__init__()
        self.decoder = decoder
        self.beam_size = beam_size
        self.sos_id = decoder.sos_id
        self.pad_id = decoder.pad_id
        self.eos_id = decoder.eos_id
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.forward_step = decoder.forward_step

    def _inflate(self, tensor: torch.Tensor, n_repeat: int, dim: int) -> torch.Tensor:
        repeat_dims = [1] * len(tensor.size())
        repeat_dims[dim] *= n_repeat
        return tensor.repeat(*repeat_dims)

    def _get_successor(
        self,
        current_ps: torch.Tensor,
        current_vs: torch.Tensor,
        finished_ids: tuple,
        num_successor: int,
        eos_count: int,
        k: int,
    ) -> int:
        finished_batch_idx, finished_idx = finished_ids

        successor_ids = current_ps.topk(k + num_successor)[1]
        successor_idx = successor_ids[finished_batch_idx, -1]

        successor_p = current_ps[finished_batch_idx, successor_idx]
        successor_v = current_vs[finished_batch_idx, successor_idx]

        prev_status_idx = successor_idx // k
        prev_status = self.ongoing_beams[finished_batch_idx, prev_status_idx]
        prev_status = prev_status.view(-1)[:-1]

        successor = torch.cat([prev_status, successor_v.view(1)])

        if int(successor_v) == self.eos_id:
            self.finished[finished_batch_idx].append(successor)
            self.finished_ps[finished_batch_idx].append(successor_p)
            eos_count = self._get_successor(
                current_ps=current_ps,
                current_vs=current_vs,
                finished_ids=finished_ids,
                num_successor=num_successor + eos_count,
                eos_count=eos_count + 1,
                k=k,
            )

        else:
            self.ongoing_beams[finished_batch_idx, finished_idx] = successor
            self.cumulative_ps[finished_batch_idx, finished_idx] = successor_p

        return eos_count

    def _get_hypothesis(self):
        predictions = list()

        for batch_idx, batch in enumerate(self.finished):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_ps[batch_idx]
                top_beam_idx = int(prob_batch.topk(1)[1])
                predictions.append(self.ongoing_beams[batch_idx, top_beam_idx])

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(self.finished_ps[batch_idx]).topk(1)[1])
                predictions.append(self.finished[batch_idx][top_beam_idx])

        predictions = self._fill_sequence(predictions)
        return predictions

    def _is_all_finished(self, k: int) -> bool:
        for done in self.finished:
            if len(done) < k:
                return False

        return True

    def _fill_sequence(self, y_hats: list) -> torch.Tensor:
        batch_size = len(y_hats)
        max_length = -1

        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.zeros((batch_size, max_length), dtype=torch.long)

        for batch_idx, y_hat in enumerate(y_hats):
            matched[batch_idx, : len(y_hat)] = y_hat
            matched[batch_idx, len(y_hat) :] = int(self.pad_id)

        return matched

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
class BeamSearchLSTM(OpenspeechBeamSearchBase):
    r"""
    LSTM Beam Search Decoder

    Args: decoder, beam_size, batch_size
        decoder (DecoderLSTM): base decoder of lstm model.
        beam_size (int): size of beam.

    Inputs: encoder_outputs, targets, encoder_output_lengths, teacher_forcing_ratio
        encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        encoder_output_lengths (torch.LongTensor): A encoder output lengths sequence. `LongTensor` of size ``(batch)``
        teacher_forcing_ratio (float): Ratio of teacher forcing.

    Returns:
        * logits (torch.FloatTensor): Log probability of model predictions.
    """

    def __init__(self, decoder: LSTMAttentionDecoder, beam_size: int):
        super(BeamSearchLSTM, self).__init__(decoder, beam_size)
        self.hidden_state_dim = decoder.hidden_state_dim
        self.num_layers = decoder.num_layers
        self.validate_args = decoder.validate_args

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_output_lengths: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Beam search decoding.

        Inputs: encoder_outputs
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        batch_size, hidden_states = encoder_outputs.size(0), None

        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]

        inputs, batch_size, max_length = self.validate_args(None, encoder_outputs, teacher_forcing_ratio=0.0)

        step_outputs, hidden_states, attn = self.forward_step(inputs, hidden_states, encoder_outputs)
        self.cumulative_ps, self.ongoing_beams = step_outputs.topk(self.beam_size)

        self.ongoing_beams = self.ongoing_beams.view(batch_size * self.beam_size, 1)
        self.cumulative_ps = self.cumulative_ps.view(batch_size * self.beam_size, 1)

        input_var = self.ongoing_beams

        encoder_dim = encoder_outputs.size(2)
        encoder_outputs = self._inflate(encoder_outputs, self.beam_size, dim=0)
        encoder_outputs = encoder_outputs.view(self.beam_size, batch_size, -1, encoder_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(batch_size * self.beam_size, -1, encoder_dim)

        if attn is not None:
            attn = self._inflate(attn, self.beam_size, dim=0)

        if isinstance(hidden_states, tuple):
            hidden_states = tuple([self._inflate(h, self.beam_size, 1) for h in hidden_states])
        else:
            hidden_states = self._inflate(hidden_states, self.beam_size, 1)

        for di in range(max_length - 1):
            if self._is_all_finished(self.beam_size):
                break

            if isinstance(hidden_states, tuple):
                tuple(
                    h.view(self.num_layers, batch_size * self.beam_size, self.hidden_state_dim) for h in hidden_states
                )
            else:
                hidden_states = hidden_states.view(self.num_layers, batch_size * self.beam_size, self.hidden_state_dim)
            step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs, attn)

            step_outputs = step_outputs.view(batch_size, self.beam_size, -1)
            current_ps, current_vs = step_outputs.topk(self.beam_size)

            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)

            current_ps = (current_ps.permute(0, 2, 1) + self.cumulative_ps.unsqueeze(1)).permute(0, 2, 1)
            current_ps = current_ps.view(batch_size, self.beam_size**2)
            current_vs = current_vs.view(batch_size, self.beam_size**2)

            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)

            topk_current_ps, topk_status_ids = current_ps.topk(self.beam_size)
            prev_status_ids = topk_status_ids // self.beam_size

            topk_current_vs = torch.zeros((batch_size, self.beam_size), dtype=torch.long).cuda()
            prev_status = torch.zeros(self.ongoing_beams.size(), dtype=torch.long).cuda()

            for batch_idx, batch in enumerate(topk_status_ids):
                for idx, topk_status_idx in enumerate(batch):
                    topk_current_vs[batch_idx, idx] = current_vs[batch_idx, topk_status_idx]
                    prev_status[batch_idx, idx] = self.ongoing_beams[batch_idx, prev_status_ids[batch_idx, idx]]

            self.ongoing_beams = torch.cat([prev_status, topk_current_vs.unsqueeze(2)], dim=2)
            self.cumulative_ps = topk_current_ps

            if torch.any(topk_current_vs == self.eos_id):
                finished_ids = torch.where(topk_current_vs == self.eos_id)
                num_successors = [1] * batch_size

                for (batch_idx, idx) in zip(*finished_ids):
                    self.finished[batch_idx].append(self.ongoing_beams[batch_idx, idx])
                    self.finished_ps[batch_idx].append(self.cumulative_ps[batch_idx, idx])

                    if self.beam_size != 1:
                        eos_count = self._get_successor(
                            current_ps=current_ps,
                            current_vs=current_vs,
                            finished_ids=(batch_idx, idx),
                            num_successor=num_successors[batch_idx],
                            eos_count=1,
                            k=self.beam_size,
                        )
                        num_successors[batch_idx] += eos_count

            input_var = self.ongoing_beams[:, :, -1]
            input_var = input_var.view(batch_size * self.beam_size, -1)

        return self._get_hypothesis()