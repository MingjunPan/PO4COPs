
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

        self.entropy = torch.zeros(1)
    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        problem_size = state.ninf_mask.size(2)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            selected = selected % problem_size
            # selected = state.START_IDX
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        elif not self.training and state.selected_count < 3 and pomo_size > problem_size:
            # import ipdb; ipdb.set_trace()
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            candidate_number = pomo_size // problem_size
            candidates = torch.topk(probs, candidate_number, dim=2)[1]
            # shape: (batch, pomo, candidate_num)
            index = torch.arange(candidate_number)[None, None, :].expand(batch_size, problem_size, candidate_number) \
                                    .transpose(1, 2).reshape(batch_size, pomo_size)
            
            selected = candidates[state.BATCH_IDX, state.POMO_IDX, index]
            # shape: (batch, pomo)
            prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            self.store_entropy(probs)
            # import ipdb; ipdb.set_trace()
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                .reshape(batch_size, pomo_size)

        return selected, prob

    def route_forward(self, route_info):
        route = route_info.route
        # shape: (batch, pomo, problem)
        batch_size = route.size(0)
        pomo_size = route.size(1)
        problem_size = route.size(2)

        start = route[:, :, 0]
        encoded_first_node = _get_encoding(self.encoded_nodes, start)
        # shape: (batch, pomo, embedding)
        self.decoder.set_q1(encoded_first_node, problem_size - 1)

        encoded_route_node = _get_encoding(self.encoded_nodes, route[:, :, :-1])
        # batch, pomo, dim
        # batch, pomo * (), dim
        probs = self.decoder(encoded_route_node, ninf_mask=route_info.ninf_mask)
        # shape: (batch, pomo * (problem - 1), problem)

        probs = probs.view(batch_size, pomo_size, problem_size - 1, problem_size)
        node_index_to_pick = route[:, :, 1:, None]
        prob = probs.gather(3, node_index_to_pick).squeeze(3)
        # shape: (batch, pomo, problem - 1)
        prob = torch.cat([torch.ones(batch_size, pomo_size, 1), prob], dim=2)
        return prob
    @torch.no_grad
    def store_entropy(self, probs):
        self.entropy = self.entropy - (probs.detach() * torch.log(probs+ 1e-10)).sum(dim=2).mean()

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo, *)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    index_shape = node_index_to_pick.shape
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[..., None].expand(*index_shape, embedding_dim).contiguous()
    # shape: (batch, pomo, *, embedding)

    if len(index_shape) == 3:
        gathering_index = gathering_index.view(batch_size, -1, embedding_dim)
        # shape: (batch, pomo * （problem - 1), embedding)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, *, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            # import ipdb; ipdb.set_trace()
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        decoder_layer_num = self.model_params.pop('decoder_layer_num', 1)

        self.layers = nn.ModuleList([DecoderLayer(mode='feature', **model_params) for _ in range(decoder_layer_num)])
        self.layers.append(DecoderLayer(mode='logit', **model_params))

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        for layer in self.layers:
            layer.set_kv(encoded_nodes)

    def set_q1(self, encoded_q1, expand=None):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.layers[0].set_q1(encoded_q1, expand)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)

        out = encoded_last_node
        for layer in self.layers:
            out = layer(out, ninf_mask)

        return out

class DecoderLayer(nn.Module):
    def __init__(self, mode = 'feature', **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = mode
        self.first = False
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        if mode == 'feature':
            self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
            self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
            self.feedForward = Feed_Forward_Module(**model_params)
            self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
        elif mode == 'logit':
            self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.Wlogit_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None
        self.v = None
        self.logitk = None
        self.q_first = None

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        if self.mode == 'feature':
            self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
            self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
            # shape: (batch, head_num, problem, qkv_dim)
        elif self.mode == 'logit':
            self.logitk = self.Wlogit_k(encoded_nodes)
            # shape: (batch, problem, embedding)

    def set_q1(self, encoded_q1, expand=None):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        batch_size = encoded_q1.shape[0]
        embedding_dim = encoded_q1.shape[2]
        head_num = self.model_params['head_num']
        self.first = True

        if expand is not None:
            encoded_q1 = encoded_q1.unsqueeze(2).expand(-1, -1, expand, -1) \
                            .contiguous().view(batch_size, -1, embedding_dim)
            
        if self.mode == 'feature':
            self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
            # shape: (batch, head_num, n, qkv_dim)
        else:
            self.q_first = self.Wq_first(encoded_q1)
            # shape: (batch, n, embedding)
    def forward(self, input, ninf_mask=None):
        # input.shape: (batch, pomo, EMBEDDING_DIM)

        if self.mode == 'feature':
            head_num = self.model_params['head_num']

            q = reshape_by_heads(self.Wq_last(input), head_num=head_num)
            # q shape: (batch, HEAD_NUM, pomo, KEY_DIM)

            if self.first:
                q = q + self.q_first
            
            out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            # shape: (batch, pomo, HEAD_NUM*KEY_DIM)

            multi_head_out = self.multi_head_combine(out_concat)
            # shape: (batch, pomo, EMBEDDING_DIM)

            out1 = self.addAndNormalization1(input, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)

            return out3
            # shape: (batch, pomo, EMBEDDING_DIM)
        elif self.mode == 'logit':
            q = self.Wq_last(input)
            # shape: (batch, pomo, embedding)

            if self.first:
                q = q + self.q_first

            score = torch.matmul(q, self.logitk.transpose(1, 2))
            # shape: (batch, pomo, problem)

            sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
            logit_clipping = self.model_params['logit_clipping']

            score_scaled = score / sqrt_embedding_dim
            # shape: (batch, pomo, problem)

            score_clipped = logit_clipping * torch.tanh(score_scaled)

            score_masked = score_clipped + ninf_mask

            probs = F.softmax(score_masked, dim=2)
            # shape: (batch, pomo, problem)

            return probs



########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
