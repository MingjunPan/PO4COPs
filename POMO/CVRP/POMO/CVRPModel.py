
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)


        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            # import ipdb; ipdb.set_trace()
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            # encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # # shape: (batch, pomo, embedding)
            # probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            # # shape: (batch, pomo, problem+1)

            # candidate_number = int(pomo_size * self.model_params['candidate_ratio'])
            # candidates = probs.topk(candidate_number, dim=2)[1]
            # # shape: (batch, pomo, candidate_number)
            # # sample from candidates for all pomo
            # index = torch.randint(0, candidate_number, size=(batch_size, pomo_size))
            # selected = candidates[state.BATCH_IDX, state.POMO_IDX, index]
            # prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] + EPS
            
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob
    def route_forward(self, route_info):
        route = route_info.route
        load = route_info.load
        # shape: (batch, pomo, seq_len)
        ninf_mask = route_info.ninf_mask
        
        batch_size = route.size(0)
        pomo_size = route.size(1)
        seq_len = route.size(2)
        problem_size = ninf_mask.size(2)

        # start = route[:, :, 0]
        # encoded_first_node = _get_encoding(self.encoded_nodes, start)
        # # shape: (batch, pomo, embedding)
        # self.decoder.set_q1(encoded_first_node, problem_size - 1)

        encoded_route_node = _get_encoding(self.encoded_nodes, route)
        # batch, pomo, dim
        # batch, pomo * seq_len, dim
        load = load.view(batch_size, -1)
        probs = self.decoder(encoded_route_node, load=load, ninf_mask=ninf_mask)
        # shape: (batch, pomo * seq_len, problem)
        probs = probs.view(batch_size, pomo_size, seq_len, problem_size)
        probs = probs[:, :, 1:-1, :]
        node_index_to_pick = route[:, :, 2:, None]
        prob = probs.gather(3, node_index_to_pick).squeeze(3)
        # shape: (batch, pomo, problem - 1)
        prob = torch.cat([torch.ones(batch_size, pomo_size, 2), prob], dim=2)
        return prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo, *)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    index_shape = node_index_to_pick.shape
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[..., None].expand(*index_shape, embedding_dim)
    # shape: (batch, pomo, *, embedding)

    if len(index_shape) == 3:
        gathering_index = gathering_index.reshape(batch_size, -1, embedding_dim)
        # shape: (batch, pomo * （problem - 1), embedding)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, *, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


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

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
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

    def forward(self, encoded_last_node, load, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)

        out = self.layers[0](encoded_last_node, load, ninf_mask)
        for layer in self.layers[1:]:
            out = layer(out, ninf_mask=ninf_mask)

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
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
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
    def forward(self, input, load=None, ninf_mask=None):
        # input.shape: (batch, pomo, EMBEDDING_DIM)

        if self.mode == 'feature':
            head_num = self.model_params['head_num']
            input_cat = torch.cat((input, load[:, :, None]), dim=2)
            
            q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
            # q shape: (batch, HEAD_NUM, pomo, KEY_DIM)

            # if self.first:
            #     q = q + self.q_first
            
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

            # if self.first:
            #     q = q + self.q_first

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


class AddAndInstanceNormalization(nn.Module):
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