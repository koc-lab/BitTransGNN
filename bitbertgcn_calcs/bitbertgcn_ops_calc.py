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

def gcn_params(dh, dg, do):
    return dh*dg+dg + dg*do+do

def gcn_add_ops(num_nodes, num_edges, dh, dg, do):
    return num_edges*dh + num_nodes*dh*(dg-1) + num_nodes*dg*(do-1) + num_nodes*do

def gcn_mult_ops(num_nodes, num_edges, dh, dg, do):
    return num_edges*dh + num_nodes*dh*dg + num_nodes*dg*do

def gcn_full_ops(num_nodes, num_edges, dh, dg, do):
    add_op = num_edges*dh + num_nodes*dh*(dg-1) + num_nodes*dg*(do-1) + num_nodes*do
    mult_op = num_edges*dh + num_nodes*dh*dg + num_nodes*dg*do
    return add_op, mult_op

def gcn_quant_ops(num_nodes, num_edges, dh, dg, do, bin):
    if bin:
        add_ops = num_edges*dh + num_nodes*dh*(2*dg-1) + num_nodes*dg*(2*do-1)
    else:
        add_ops = num_edges*dh + num_nodes*dh*(3*dg-1) + num_nodes*dg*(3*do-1)
    mult_ops = num_edges*dh
    return add_ops, mult_ops

def gcn_ops(num_nodes, num_edges, dh, dg, do, bits):
    if bits == 32.0:
        add, mult = gcn_full_ops(num_nodes, num_edges, dh, dg, do)
    else:
        if bits == 1.00 or bits == 1.58:
            bin = True
        elif bits == 2.00 or bits == 2.32:
            bin = False
        add, mult = gcn_quant_ops(num_nodes, num_edges, dh, dg, do, bin)
    return add, mult

def bert_add_ops(dh, M, L, H, do):
    ds = dh/12
    df = dh*4
    emb = 2*M*dh
    attn = M*L*H*3*dh*ds+M*dh*dh + 2*L*H*(M-1)*M*ds
    ffn = M*dh*df+M*df*dh
    pool = M*dh*dh
    clsif = M*dh*do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_quant_add_ops(dh, M, L, H, do, bin=True):
    if bin:
        add_factor = 2
    else:
        add_factor = 3
    ds = dh/12
    df = dh*4
    emb = 2*M*dh
    attn = M*L*H*3*(add_factor*dh-1)*ds + M*(add_factor*dh-1)*dh + 2*L*H*(M-1)*M*ds
    ffn = M*(add_factor*dh-1)*df + M*(add_factor*df-1)*dh
    pool = M*(add_factor*dh-1)*dh
    clsif = M*(add_factor*dh-1)*do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_mult_ops(dh, M, L, H, do):
    ds = dh/12
    df = dh*4
    emb = 2*M*dh
    attn = M*L*H*3*dh*ds+M*dh*dh + 2*L*H*M*M*ds
    ffn = M*dh*df + M*df*dh
    pool = M*dh*dh
    clsif = M*dh*do
    total = emb + attn + ffn + pool + clsif
    return total

def bert_quant_mult_ops(dh, M, L, H, do):
    ds = dh/12
    df = dh*4
    emb = 2*M*dh
    attn = M*L*H*3*(dh+ds)+M*(dh+dh) + 2*L*H*M*M*ds
    ffn = M*(dh+df) + M*(df+dh)
    pool = M*(dh+dh)
    clsif = M*(dh+do)
    total = emb + attn + ffn + pool + clsif
    return total

def bert_ops(model_type, model_size, bits):
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
    return add, mult

def bert_ops_info(model_type, model_size, exact=False, inv=False):
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
