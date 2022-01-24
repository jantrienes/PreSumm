from src.prepro.data_builder import BertData


def test_preprocess():
    class args:
        pretrained_model = "bert-base-uncased"
        lower = True
        min_src_nsents = 1
        max_src_nsents = 200
        min_src_ntokens_per_sent = 3
        max_src_ntokens_per_sent = 200
        min_tgt_ntokens = 1
        max_tgt_ntokens = 200

    data = BertData(args)

    src = [
        "this is a test".split(),
        "too short".split(),
        "third sentence is long enough".split(),
        "this is the fourth sentence".split(),
    ]
    tgt = ["this is the summary".split()]
    src_sent_labels = [0, 2]

    b_data = data.preprocess(src, tgt, src_sent_labels)

    assert data.tokenizer.convert_ids_to_tokens(b_data.src_subtoken_idxs) == [
        "[CLS]",
        "this",
        "is",
        "a",
        "test",
        "[SEP]",
        "[CLS]",
        "third",
        "sentence",
        "is",
        "long",
        "enough",
        "[SEP]",
        "[CLS]",
        "this",
        "is",
        "the",
        "fourth",
        "sentence",
        "[SEP]",
    ]
    assert b_data.sent_labels == [1, 1, 0]
    assert data.tokenizer.convert_ids_to_tokens(b_data.tgt_subtoken_idxs) == [
        "[unused9]",
        "this",
        "is",
        "the",
        "summary",
        "[unused1]",
    ]
    assert b_data.segments_ids == [0] * 6 + [1] * 7 + [0] * 7
    assert b_data.cls_ids == [0, 6, 13]
    assert b_data.src_txt == [
        "this is a test",
        "third sentence is long enough",
        "this is the fourth sentence",
    ]
    assert b_data.tgt_txt == "this is the summary"
    assert b_data.original_idxs == [0, 2, 3]  # original indices of selected sentences
