"""
This module contains layers that hold memories on activation. Layers in this
module are most similar to the layers in the module synapse_layers. Layers in
synapse layers learn weights, which are dual to activation, to form
hierarchical patterns, while layers in memory_layers hold memory for activation
in one layer (though it is possible the memory can be hierarchically used some
day).

The above description could be confusing, since it is not something that is
widely accepted in the deep learning community, but an unpublished, and
untested hypothesis made by me. So you may want to look at what the layers in
this module actually do, and learn how they are used in the community.
"""

from ..core.blocks import ProcessingLayer
from .. import backend as A

class MemoryLayer(ProcessingLayer):
    pass


class EmbeddingLayer(MemoryLayer):
    NAME = "Embedding"
    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super(EmbeddingLayer, self).__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _setup(self):
        self.embeddings = self._get_variable("embeddings",
                                             [self.num_embeddings, self.embedding_dim],
                                             self._get_initializer())

    def _forward(self, idxs):
        self._data = A.nn.embedding(idxs, self.embeddings)
        return self._data

Embedding = EmbeddingLayer
