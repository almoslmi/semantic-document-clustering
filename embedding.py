import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from typing import List


tf.logging.set_verbosity(tf.logging.ERROR)


class VarTextEmbedder:

    def __init__(self):
        self.embeder = hub.Module(
            "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")

        self.session = tf.Session()
        self.session.run(
            [tf.global_variables_initializer(), tf.tables_initializer()]
        )

    def embed(self, vartext: np.ndarray) -> np.ndarray:
        return self.session.run(self.embeder(vartext))

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> 'VarTextEmbedder':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
