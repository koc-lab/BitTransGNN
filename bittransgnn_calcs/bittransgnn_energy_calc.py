from bittransgnn_ops_calc import bert_ops, gnn_ops
from typing import Optional

def tot_energy(addop, multop, dtype, processor):
    if dtype == "int32":
        if processor == "7nm":
            addeng = 0.03e-12
            multeng = 1.48e-12
        elif processor == "45nm":
            addeng = 0.1e-12
            multeng = 3.1e-12
    elif dtype == "int8":
        if processor == "7nm":
            addeng = 0.007e-12
            multeng = 0.07e-12
        elif processor == "45nm":
            addeng = 0.03e-12
            multeng = 0.2e-12
    elif dtype == "float32":
        if processor == "7nm":
            addeng = 0.38e-12
            multeng = 1.31e-12
        elif processor == "45nm":
            addeng = 0.9e-12
            multeng = 3.7e-12
    return addeng*addop, multeng*multop

def bittransformer_energy_info(model_type, model_size, exact=True, inv=False, dataset_conf: Optional[dict] = None, full_batch=False):
    highbit = 32.0
    bit_list = [32.0, 1.0, 2.0]
    #processor_list = ["7nm", "45nm"]
    processor_list = ["7nm"]
    for processor in processor_list:
        for bits in bit_list:
            if bits == 32.0:
                dtype_list = ["float32"]
            else:
                dtype_list = ["int8", "int32", "float32"]
            
            addop_high, multop_high = bert_ops(model_type, model_size, highbit, dataset_conf, full_batch)
            add_energy_high, mult_energy_high = tot_energy(addop_high, multop_high, "float32", processor)
            total_energy_high = add_energy_high + mult_energy_high
            for dtype in dtype_list:
                print("----------")
                print("bits: ", bits)
                print("dtype: ", dtype)
                print("processor: ", processor)
                addop, multop = bert_ops(model_type, model_size, bits, dataset_conf, full_batch)
                add_energy, mult_energy = tot_energy(addop, multop, dtype, processor)
                total_energy = add_energy + mult_energy
                if bits == highbit:
                    print("add energy: ", add_energy)
                    print("mult energy: ", mult_energy)
                    print("total energy: ", total_energy)
                else:             
                    if exact:
                        print("add energy: ", add_energy)
                        print("mult energy: ", mult_energy)
                        print("total energy: ", total_energy)
                    else:
                        if inv:
                            print("add energy: ", add_energy_high/add_energy)
                            print("mult energy: ", mult_energy_high/mult_energy)
                            print("total energy: ", total_energy_high/total_energy)
                        else:
                            print("add energy: ", add_energy/add_energy_high)
                            print("mult energy: ", mult_energy/mult_energy_high)
                            print("total energy: ", total_energy/total_energy_high)

def bittransgnn_energy_info(model_type, model_size, gnn_conf: dict, exact=True, inv=False, dataset_conf: Optional[dict] = None, full_batch=True, batch_size=32, train_type="static", gnn_bits=32):
    highbit = 32.0
    bit_list = [32.0, 1.0, 2.0]
    processor_list = ["7nm"]
    for processor in processor_list:
        for bits in bit_list:
            if bits == 32.0:
                dtype_list = ["float32"]
            else:
                dtype_list = ["int8", "int32", "float32"]
            addop_high, multop_high = bert_ops(model_type, model_size, highbit, dataset_conf, full_batch)
            add_energy_high, mult_energy_high = tot_energy(addop_high, multop_high, "float32", processor)
            total_energy_high = add_energy_high + mult_energy_high
            #if full_batch:
            #    add_energy_high, mult_energy_high, total_energy_high = add_energy_high * num_sequences, mult_energy_high * num_sequences, total_energy_high * num_sequences
            for dtype in dtype_list:
                print("----------")
                print("bits: ", bits)
                print("dtype: ", dtype)
                print("processor: ", processor)
                addop, multop = bert_ops(model_type, model_size, bits, dataset_conf, full_batch)
                add_energy, mult_energy = tot_energy(addop, multop, dtype, processor)
                total_energy = add_energy + mult_energy
                #if full_batch:
                #    add_energy, mult_energy, total_energy = add_energy * num_sequences, mult_energy * num_sequences, total_energy * num_sequences
                num_nodes, num_edges, dh, dg, do, gnn_bits = gnn_conf["num_nodes"], gnn_conf["num_edges"], gnn_conf["dh"], gnn_conf["dg"], gnn_conf["do"], gnn_conf["gnn_bits"]
                gnn_add_ops, gnn_mult_ops = gnn_ops(num_nodes, num_edges, dh, dg, do, bits=gnn_bits, dataset_conf=dataset_conf, train_type=train_type, batch_size=batch_size)
                gnn_add_energy, gnn_mult_energy = tot_energy(gnn_add_ops, gnn_mult_ops, dtype, processor)
                gnn_energy = gnn_add_energy + gnn_mult_energy
                #if train_type == "dynamic":
                #    gnn_add_energy, gnn_mult_energy = gnn_add_energy * num_sequences / batch_size, gnn_mult_energy * num_sequences / batch_size
                #    gnn_energy = gnn_energy * num_sequences / batch_size
                total_energy += gnn_energy
                add_energy += gnn_add_energy
                mult_energy += gnn_mult_energy
                #if bits == highbit:
                #    print("add energy: ", add_energy)
                #    print("mult energy: ", mult_energy)
                #    print("total energy: ", total_energy)
                if exact:
                    print("add energy: ", add_energy)
                    print("mult energy: ", mult_energy)
                    print("total energy: ", total_energy)
                else:
                    if inv:
                        print("add energy: ", add_energy_high/add_energy)
                        print("mult energy: ", mult_energy_high/mult_energy)
                        print("total energy: ", total_energy_high/total_energy)
                    else:
                        print("add energy: ", add_energy/add_energy_high)
                        print("mult energy: ", mult_energy/mult_energy_high)
                        print("total energy: ", total_energy/total_energy_high)


def get_bittransformer_total_energy(model_type, model_size, bits, dtype, dataset_conf: Optional[dict] = None, processor = "7nm", exact=True, inv=False, add_gnn=False, gnn_conf: Optional[dict] = None, full_batch=False, train_type="static", batch_size=32):
    if add_gnn:
        assert (gnn_conf is not None)
    addop, multop = bert_ops(model_type, model_size, bits, dataset_conf, full_batch)
    add_energy, mult_energy = tot_energy(addop, multop, dtype, processor)
    total_energy = add_energy + mult_energy
    if add_gnn:
        num_nodes, num_edges, dh, dg, do, gnn_bits = gnn_conf["num_nodes"], gnn_conf["num_edges"], gnn_conf["dh"], gnn_conf["dg"], gnn_conf["do"], gnn_conf["gnn_bits"]
        #gnn_add_ops, gnn_mult_ops = gnn_ops(num_nodes, num_edges, dh, dg, do)
        #gnn_add_ops, gnn_mult_ops = gnn_quant_ops(num_nodes, num_edges, dh, dg, do)
        gnn_add_ops, gnn_mult_ops = gnn_ops(num_nodes, num_edges, dh, dg, do, bits=gnn_bits, dataset_conf=dataset_conf, train_type=train_type, batch_size=batch_size)
        gnn_add_energy, gnn_mult_energy = tot_energy(gnn_add_ops, gnn_mult_ops, dtype, processor)
        gnn_energy = gnn_add_energy + gnn_mult_energy
        total_energy += gnn_energy
    return total_energy
