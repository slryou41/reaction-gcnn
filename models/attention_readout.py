import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class AttentionReadout(chainer.Chain):
    """GGNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector associated to
            each atom
        nobias (bool): If ``True``, then this function does not use
            the bias
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function for node representation
            `functions.tanh` was suggested in original paper.
        activation_agg (~chainer.Function or ~chainer.FunctionNode):
            activate function for aggregation
            `functions.tanh` was suggested in original paper.
    """

    def __init__(self, out_dim, hidden_dim=16, nobias=False,
                 activation=functions.identity,
                 activation_agg=functions.identity,
                 vis_attention=False):
        super(AttentionReadout, self).__init__()
        with self.init_scope():
            self.i_layer = GraphLinear(None, out_dim, nobias=nobias)
            self.j_layer = GraphLinear(None, out_dim, nobias=nobias)
            
            self.ch_attention = GraphLinear(None, out_dim*3, nobias=nobias)
            self.ch_layer = GraphLinear(None, out_dim*3, nobias=nobias) 
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.nobias = nobias
        self.activation = activation
        self.activation_agg = activation_agg
        
        self.vis_attention = vis_attention

    def __call__(self, h, h0=None, is_real_node=None):
        # --- Readout part ---
        # h, h0: (minibatch, node, ch)
        # is_real_node: (minibatch, node)
        gs = []
        for ii in range(len(h)):
            # h1 = functions.concat((h[ii], h0[ii]), axis=2) if h0[ii] is not None else h[ii]
            h1 = h[ii]

            g1 = functions.sigmoid(self.i_layer(h1))
            g2 = self.activation(self.j_layer(h1))
            g = g1 * g2
            if is_real_node is not None:
            # if is_real_node[ii] is not None:
                # mask virtual node feature to be 0
                mask = self.xp.broadcast_to(
                    is_real_node[ii][:, :, None], g.shape)
                g = g * mask
            gs.append(g)
            
        _g = functions.concat((gs[0], gs[1], gs[2]), axis=1)
        att = functions.sigmoid(self.ch_attention(_g))
        final_g = self.activation(self.ch_layer(_g))
        
        g = att * final_g
        _g = g

        # sum along node axis
        g = self.activation_agg(functions.sum(g, axis=1))
        
        # if self.vis_attention:
        #     return g, att
        
        return g #, _g, att
