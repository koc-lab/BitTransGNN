from bitbertgcn_ops_calc import bert_embedding_params, bert_attn_params, bert_ffn_params, bert_pooler_params, bert_layernorms, bert_clsif, gcn_params

def bert_bytes(model_type, model_size, be, bl, add_clsif, bg=32.0, do=20, add_gcn=False):
    if model_type == "bert":
        V = 30522
        S = 512
        T = 2
    elif model_type == "roberta":
        V = 50265
        S = 514
        T = 1
    
    if model_size == "base":
        dh = 768
        H = 12
        L = 12
    elif model_size == "large":
        dh = 1024
        H = 16
        L = 24
    bfull = 32/8
    be = be/8
    bl = bl/8
    dg = 256
    #do = 20
    emb = bert_embedding_params(V,S,T,dh)
    attn = bert_attn_params(L, H, dh)
    ffn = bert_ffn_params(L, dh)
    pool = bert_pooler_params(dh)
    layernorms = bert_layernorms(L, dh)
    clsif = bert_clsif(dh)
    gcn_param = gcn_params(dh, dg, do)
    total_bits = be*emb + bl*(attn+ffn+pool) + bfull*layernorms
    if add_clsif:
        total_bits += bl*clsif
    if add_gcn:
        total_bits += bg*gcn_param
    return total_bits

def return_bert_byte_info(model_type, model_size, exact=False, add_clsif=False):
    print(f"{model_type}-{model_size}")
    highbit = 32.0
    lowbit_list = [1.0, 1.58, 2.0, 2.32]
    full_prec = bert_bytes(model_type, model_size, highbit, highbit, add_clsif)
    print("-----------")
    print(f"w{highbit}e{highbit}")
    print(full_prec)
    print("-----------")
    for lowbit in lowbit_list:
        bit_list = [highbit, lowbit]
        for be in bit_list:
            for bl in bit_list:
                if not(be == 32.0 and bl == 32.0):
                    print(f"w{bl}e{be}")
                    if exact:
                        print(bert_bytes(model_type, model_size, be, bl, add_clsif))
                    else:
                        print(bert_bytes(model_type, model_size, be, bl, add_clsif)/full_prec)
        print("-----------")

def return_bertgcn_byte_info(model_type, model_size, exact=False, add_clsif=True, quantize_embeddings=False):
    if quantize_embeddings:
        be = 1.0
    else:
        be = 32.0
    highbit = 32.0
    bit_list = [1.0, 1.58, 2.32, 32.0]
    full_prec = bert_bytes(model_type, model_size, highbit, highbit, add_clsif, bg=highbit, add_gcn=True)
    print("-----------")
    print(f"bert{highbit}gcn{highbit}")
    print(full_prec)
    print("-----------")
    for bl in bit_list:
        for bg in bit_list:
            if not(bl == 32.0 and bg == 32.0):
                print(f"bert{bl}gcn{bg}")
                if exact:
                    print(bert_bytes(model_type, model_size, be, bl, add_clsif, bg=bg, add_gcn=True))
                else:
                    print(bert_bytes(model_type, model_size, be, bl, add_clsif, bg=bg, add_gcn=True)/full_prec)
        print("-----------")