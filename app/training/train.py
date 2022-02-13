import time


def run_epoch(data_iteration, model, loss_c):
    s_time = time.time()

    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iteration):
        m_out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_c(m_out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elp_time = time.time() - s_time
            print(
                "Epoch Step: {} Loss: {} Tokens per Sec: {}".format(
                    i, loss / batch.ntokens, tokens / elp_time
                )
            )
            s_time = time.time()
            tokens = 0

    return total_loss / total_tokens


def batch_size_fun(new, count):
    if count == 1:
        max_src_in_batch = 0
        max_trg_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_trg_in_batch = max(max_trg_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch
    return max(src_elements, trg_elements), max_src_in_batch, max_trg_in_batch
