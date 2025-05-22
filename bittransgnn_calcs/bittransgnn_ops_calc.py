from typing import Optional

def bert_embedding_params(V, S, T, dh):
    return V*dh+S*dh+T*dh

def bert_attn_params(L, H, dh):
    ds = dh/12
    return L*(H*3*(dh*ds+ds)+(dh*dh+dh))

def bert_ffn_params(L, dh):
    df = dh*4
    return L*((dh*df+df) + (df*dh+dh))

def bert_pooler_params(dh):
    return dh*dh+dh

def bert_layernorms(L, dh):
    return L*(dh+dh) + L*(dh+dh) + (dh+dh)

def bert_clsif(dh, do=20):
    return dh*do+do

def gnn_params(dh, dg, do):
    return dh*dg+dg + dg*do+do

def gnn_full_ops(num_nodes, num_edges, dh, dg, do):
    #add_op, mult_op = gnn_add_ops(num_nodes, edge_count, dh, dg, do), gnn_mult_ops(num_nodes, edge_count, dh, dg, do)
    add_op = 2*num_edges*(dh-1) + num_nodes*dh*(dg-1) + num_nodes*dg + num_nodes*dg*(do-1) + num_nodes*do
    mult_op = 2*num_edges*dh + num_nodes*dh*dg + num_nodes*dg*do
    return add_op, mult_op

def gnn_quant_ops(num_nodes, num_edges, dh, dg, do, bin):
    if bin:
        add_ops = 2*num_edges*(dh-1) + num_nodes*dh*(2*dg-1) + num_nodes*dg*(2*do-1)
    else:
        add_ops = 2*num_edges*(dh-1) + num_nodes*dh*(3*dg-1) + num_nodes*dg*(3*do-1)
    mult_ops = num_edges*dh
    return add_ops, mult_ops

def gnn_ops(num_nodes, num_edges, dh, dg, do, bits, dataset_conf: Optional[dict] = None, train_type="static", batch_size=32):
    if bits == 32.0:
        add, mult = gnn_full_ops(num_nodes, num_edges, dh, dg, do)
    else:
        if bits == 1.00 or bits == 1.58:
            bin = True
        elif bits == 2.00 or bits == 2.32:
            bin = False
        add, mult = gnn_quant_ops(num_nodes, num_edges, dh, dg, do, bin)
    if train_type == "dynamic":
        assert (dataset_conf is not None)
        num_sequences = dataset_conf["num_sequences"]
        add, mult = add*num_sequences/batch_size, mult*num_sequences/batch_size
    return add, mult

def bert_add_ops(dh, M, L, H, do):
    ds = dh/H
    df = dh*4
    emb = 2*M*dh
    attn = L*(M*H*3*dh*ds + M*dh*dh + 2*H*(M-1)*M*ds + 2*M*dh)
    ffn = L*(M*dh*df+M*df*dh + 2*M*dh)
    #pool = M*dh*dh
    #clsif = M*dh*do
    pool = dh*dh
    clsif = dh*do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_quant_add_ops(dh, M, L, H, do, bin=True):
    if bin:
        add_factor = 2
    else:
        add_factor = 3
    ds = dh/H
    df = dh*4
    emb = 2*M*dh
    attn = L*(M*H*3*(add_factor*dh-1)*ds + M*(add_factor*dh-1)*dh + 2*H*(M-1)*M*ds + 2*M*dh)
    ffn = L*(M*(add_factor*dh-1)*df + M*(add_factor*df-1)*dh + 2*M*dh)
    #pool = M*(add_factor*dh-1)*dh
    #clsif = M*(add_factor*dh-1)*do
    pool = (add_factor*dh-1)*dh
    clsif = (add_factor*dh-1)*do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_mult_ops(dh, M, L, H, do):
    ds = dh/H
    df = dh*4
    emb = 2*M*dh
    attn = L*(M*H*3*dh*ds + M*dh*dh + 2*H*M*M*ds + 2*M*dh)
    ffn = L*(M*dh*df + M*df*dh + 2*M*dh)
    #pool = M*dh*dh
    #clsif = M*dh*do
    pool = dh*dh
    clsif = dh*do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_quant_mult_ops(dh, M, L, H, do): #final version (dequantizes the output through a single step)
    ds = dh/H
    df = dh*4
    emb = 2*M*dh
    attn = L*(H*3*M*ds + M*dh + 2*H*M*M*ds + 2*M*dh)
    ffn = L * (M*df + M*dh + 2*M*dh)
    #pool = dh*(M+dh)
    #clsif = dh*(M+do)
    pool = dh
    clsif = do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_ops(model_type, model_size, bits, dataset_conf: Optional[dict] = None, full_batch = False):
    M = 128
    do = 20
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

    if bits == 32.0:
        add = bert_add_ops(dh, M, L, H, do)
        mult = bert_mult_ops(dh, M, L, H, do)
    else:
        if bits == 1.00 or bits == 1.58:
            bin = True
        elif bits == 2.00 or bits == 2.32:
            bin = False
        add = bert_quant_add_ops(dh, M, L, H, do, bin)
        mult = bert_quant_mult_ops(dh, M, L, H, do)
    if full_batch:
        assert (dataset_conf is not None)
        num_sequences = dataset_conf["num_sequences"]
        add, mult = add*num_sequences, mult*num_sequences
    return add, mult

def bert_ops_info(model_type, model_size, exact=False, inv=False, dataset_conf: Optional[dict] = None, full_batch=False):
    highbit = 32.0
    model_type = "bert"
    bits_list = [32.0, 1.0, 2.0]
    for bits in bits_list:
        print("----------")
        print(bits, " bits")
        addop, multop = bert_ops(model_type, model_size, bits, dataset_conf, full_batch)
        if bits == highbit:
            print("add: ", addop)
            print("mult: ", multop)
        else:
            if exact:
                print("add: ", addop)
                print("mult: ", multop)
            else:
                highbit_addop, highbit_multop = bert_ops(model_type, model_size, highbit, dataset_conf, full_batch)
                if inv:
                    print("add: ", highbit_addop/addop)
                    print("mult: ", highbit_multop/multop)
                else:
                    print("add: ", addop/highbit_addop)
                    print("mult: ", multop/highbit_multop)

def bittransgnn_ops_info(model_type, model_size, exact=False, inv=False):
    highbit = 32.0
    model_type = "bert"
    bits_list = [32.0, 1.0, 2.0]
    for bits in bits_list:
        print("----------")
        print(bits, " bits")
        addop, multop = bert_ops(model_type, model_size, bits)
        if bits == highbit:
            print("add: ", addop)
            print("mult: ", multop)
        else:
            if exact:
                print("add: ", addop)
                print("mult: ", multop)
            else:
                highbit_addop, highbit_multop = bert_ops(model_type, model_size, highbit)
                if inv:
                    print("add: ", highbit_addop/addop)
                    print("mult: ", highbit_multop/multop)
                else:
                    print("add: ", addop/highbit_addop)
                    print("mult: ", multop/highbit_multop)
