from model.Transformer import Transformer


def test_model_creating():
    src_vocab_size = 10
    trg_vocab_size = 10
    number_of_heads = 8
    model_dim = 512
    position_wise_dim_ff = 2048
    number_of_layers = 2

    tmp_model = Transformer(src_vocab_size,
                            trg_vocab_size,
                            number_of_heads,
                            model_dim,
                            position_wise_dim_ff,
                            number_of_layers)

    tmp_model.make_model()

if __name__ == '__main__':
    test_model_creating()