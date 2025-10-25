import os
import random
import numpy as np
import torch
from io import BytesIO
from pathlib import Path

def set_seed(seed_value=0):
    # Set Python's built-in random module seed
    random.seed(seed_value)

    # Set NumPy seed
    np.random.seed(seed_value)

    # Set PyTorch seed for CPU and CUDA
    torch.manual_seed(seed_value)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU environments
        # Ensure that cuDNN is deterministic to avoid non-deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_quant_name(num_states):
    if num_states == 2:
        quant_name = "binary"
    elif num_states == 3:
        quant_name = "ternary"
    elif num_states == 4:
        quant_name = "4_state"
    elif num_states == 5:
        quant_name = "5_state"
    elif num_states == 0:
        quant_name = "full"
    else:
        raise Exception("Invalid quantization rate entered, please enter a valid rate.")
    return quant_name

def get_train_state(joint_training):
    if joint_training:
        train_state = "dynamic"
    else:
        train_state = "static"
    return train_state

def get_model_type(quantize_bert, quantize_gcn):
    if quantize_bert and quantize_gcn:
        model_type = "bittransbitgnn"
    elif quantize_bert and not(quantize_gcn):
        model_type = "bittransgnn"
    elif not(quantize_bert) and quantize_gcn:
        model_type = "transbitgnn"
    else:
        model_type = "transgnn"
    return model_type

def set_bittrans_ckpt_dir(checkpoint_dir, quantize_bert, quantize_embeddings, quantize_attention, 
                         bert_pre_model, 
                         quant_name, dataset_name, num_bits_act,
                         adj_type="full", 
                         inference=False, direct_induction=False):
    data_path = Path(__file__).parent.parent.joinpath("./model_checkpoints")
    result_data_path = data_path.joinpath("./results")
    logit_data_path = data_path.joinpath("./logits")
    model_data_path = data_path.joinpath("./model_ckpts")
    result_data_path = result_data_path.joinpath("./bitbert")
    model_data_path = model_data_path.joinpath("./bitbert")
    logit_data_path = logit_data_path.joinpath("./bitbert")
    if inference:
        result_data_path = result_data_path.joinpath("./inference")
        model_data_path = model_data_path.joinpath("./inference")
        logit_data_path = logit_data_path.joinpath("./inference")
        use_case = "used for inference"
        if direct_induction:
            result_data_path = result_data_path.joinpath("./direct_induction")
            model_data_path = model_data_path.joinpath("./direct_induction")
            logit_data_path = logit_data_path.joinpath("./direct_induction")
            if adj_type is not None:
                result_data_path = result_data_path.joinpath(f"./{adj_type}")
                model_data_path = model_data_path.joinpath(f"./{adj_type}")
                logit_data_path = logit_data_path.joinpath(f"./{adj_type}")
            else:
                result_data_path = result_data_path.joinpath(f"./full")
                model_data_path = model_data_path.joinpath(f"./full")
                logit_data_path = logit_data_path.joinpath(f"./full")
    else:
        result_data_path = result_data_path.joinpath("./train")
        model_data_path = model_data_path.joinpath("./train")
        logit_data_path = logit_data_path.joinpath("./train")
        use_case = "fine-tuned"
    if checkpoint_dir is None:
        if quantize_bert:
            if quantize_embeddings and quantize_attention:
                print(f"A quantized transformer with quantized embeddings is being {use_case}.")
                ckpt_dir = './qat/full_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
            elif quantize_embeddings and not(quantize_attention):
                print(f"A quantized transformer with quantized embeddings and full-prec attention is being {use_case}.")
                ckpt_dir = './qat/weight_emb_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
            elif not(quantize_embeddings) and quantize_attention:
                print(f"A quantized transformer with full precision embeddings is being {use_case}.")
                ckpt_dir = './qat/weight_att_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
            else:
                print(f"A quantized transformer with full precision embeddings and full-precision attention is being {use_case}.")
                ckpt_dir = './qat/lin_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
        else:
            if quantize_embeddings:
                print(f"A full-precision transformer with quantized embeddings is being {use_case}.")
                ckpt_dir = './qat/embed_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
            else:
                print(f"A full-precision transformer is being {use_case}.")
                ckpt_dir = './full_prec/{}_{}'.format(bert_pre_model, dataset_name)
        if not(quantize_bert):
            if num_bits_act != 32.0:
                print(f"Activations are being quantized to {num_bits_act} bits.")
            else:
                print(f"Activations are kept full-precision at {num_bits_act} bits.")
        if num_bits_act not in [8.0, 32.0]:
            ckpt_dir += "_quant_act"
        elif num_bits_act == 32.0:
            ckpt_dir += "_full_act"
        #ckpt_dir = data_path.joinpath(data_path, ckpt_dir)
        result_ckpt_dir = result_data_path.joinpath(ckpt_dir)
        model_ckpt_dir = model_data_path.joinpath(ckpt_dir)
        logit_ckpt_dir = logit_data_path.joinpath(ckpt_dir)
    else:
        result_ckpt_dir = checkpoint_dir
        model_ckpt_dir = checkpoint_dir
    ckpt_dir_dict = {"result_ckpt_dir": result_ckpt_dir, "model_ckpt_dir": model_ckpt_dir, "logit_ckpt_dir": logit_ckpt_dir}
    return ckpt_dir_dict

def set_bittransgnn_ckpt_dir(checkpoint_dir, 
                            model_type, 
                            quantize_bert, quantize_embeddings, quantize_attention, bert_quant_type, 
                            bert_pre_model, 
                            train_state, quant_name, 
                            dataset_name, 
                            num_bits_act,
                            adj_type="full",
                            inference=False,
                            inference_type="transductive"):
    data_path = Path(__file__).parent.parent.joinpath("./model_checkpoints")
    result_data_path = data_path.joinpath("./results")
    model_data_path = data_path.joinpath("./model_ckpts")
    logit_data_path = data_path.joinpath("./logits")
    result_data_path = result_data_path.joinpath(f"./{model_type}")
    model_data_path = model_data_path.joinpath(f"./{model_type}")
    logit_data_path = logit_data_path.joinpath(f"./{model_type}")

    if adj_type is not None:
        result_data_path = result_data_path.joinpath(f"./{adj_type}")
        model_data_path = model_data_path.joinpath(f"./{adj_type}")
        logit_data_path = logit_data_path.joinpath(f"./{adj_type}")
    else:
        result_data_path = result_data_path.joinpath(f"./full_adj")
        model_data_path = model_data_path.joinpath(f"./full_adj")
        logit_data_path = logit_data_path.joinpath(f"./full_adj")

    if inference:
        result_data_path = result_data_path.joinpath("./inference")
        model_data_path = model_data_path.joinpath("./inference")
        logit_data_path = logit_data_path.joinpath("./inference")
    else:
        result_data_path = result_data_path.joinpath("./train")
        model_data_path = model_data_path.joinpath("./train")
        logit_data_path = logit_data_path.joinpath("./train")
    result_data_path = result_data_path.joinpath(f"./{inference_type}")
    model_data_path = model_data_path.joinpath(f"./{inference_type}")
    logit_data_path = logit_data_path.joinpath(f"./{inference_type}")
    #if distillation:
    #    data_path = data_path.joinpath("./distillation")
    if checkpoint_dir is None:
        if quantize_bert:
            if quantize_embeddings and quantize_attention:
                ckpt_dir = './full_quant/'
            elif quantize_embeddings and not(quantize_attention):
                ckpt_dir = './weight_emb_quant/'
            elif not(quantize_embeddings) and quantize_attention:
                ckpt_dir = './weight_att_quant/'
            else:
                ckpt_dir = './lin_quant/'
        else:
            if quantize_embeddings:
                ckpt_dir = './embed_quant/'
            else:
                ckpt_dir = './full_prec/'
        ckpt_dir += '{}_{}_{}_{}_{}'.format(bert_pre_model, bert_quant_type, train_state, quant_name, dataset_name)
        if not(quantize_bert):
            if num_bits_act != 32.0:
                print(f"Activations are being quantized to {num_bits_act} bits.")
            else:
                print(f"Activations are kept full-precision at {num_bits_act} bits.")
        if num_bits_act not in [8.0, 32.0]:
            ckpt_dir += "_quant_act"
        elif num_bits_act == 32.0:
            ckpt_dir += "_full_act"
        #ckpt_dir = os.path.join(data_path, ckpt_dir)
        #ckpt_dir = data_path.joinpath(ckpt_dir)
        result_ckpt_dir = result_data_path.joinpath(ckpt_dir)
        model_ckpt_dir = model_data_path.joinpath(ckpt_dir)
        logit_ckpt_dir = logit_data_path.joinpath(ckpt_dir)
    else:
        result_ckpt_dir = checkpoint_dir
        model_ckpt_dir = checkpoint_dir
    ckpt_dir_dict = {"result_ckpt_dir": result_ckpt_dir, "model_ckpt_dir": model_ckpt_dir, "logit_ckpt_dir": logit_ckpt_dir}
    return ckpt_dir_dict
            
def set_bittransgnn_kd_ckpt_dir(checkpoint_dir, 
                               model_type, 
                               quantize_bert, quantize_embeddings, quantize_attention, student_bert_quant_type,
                               bert_pre_model, 
                               train_state, student_quant_name, 
                               dataset_name, 
                               adj_type="full", 
                               num_bits_act=8.0,
                               inference=False,
                               inference_type="transductive"):
    data_path = Path(__file__).parent.parent.joinpath("./model_checkpoints")
    result_data_path = data_path.joinpath("./results")
    model_data_path = data_path.joinpath("./model_ckpts")
    logit_data_path = data_path.joinpath("./logits")
    result_data_path = result_data_path.joinpath(f"./distill/{model_type}")
    model_data_path = model_data_path.joinpath(f"./distill/{model_type}")
    logit_data_path = logit_data_path.joinpath(f"./distill/{model_type}")

    if adj_type is not None:
        result_data_path = result_data_path.joinpath(f"./{adj_type}")
        model_data_path = model_data_path.joinpath(f"./{adj_type}")
        logit_data_path = logit_data_path.joinpath(f"./{adj_type}")
    else:
        result_data_path = result_data_path.joinpath(f"./full_adj")
        model_data_path = model_data_path.joinpath(f"./full_adj")
        logit_data_path = logit_data_path.joinpath(f"./full_adj")

    if inference:
        result_data_path = result_data_path.joinpath("./inference")
        model_data_path = model_data_path.joinpath("./inference")
        logit_data_path = logit_data_path.joinpath("./inference")
    else:
        result_data_path = result_data_path.joinpath("./train")
        model_data_path = model_data_path.joinpath("./train")
        logit_data_path = logit_data_path.joinpath("./train")
    result_data_path = result_data_path.joinpath(f"./{inference_type}")
    model_data_path = model_data_path.joinpath(f"./{inference_type}")
    logit_data_path = logit_data_path.joinpath(f"./{inference_type}")
    if checkpoint_dir is None:
        if quantize_bert:
            if quantize_embeddings and quantize_attention:
                ckpt_dir = './full_quant/'
            elif quantize_embeddings and not(quantize_attention):
                ckpt_dir = './weight_emb_quant/'
            elif not(quantize_embeddings) and quantize_attention:
                ckpt_dir = './weight_att_quant/'
            else:
                ckpt_dir = './lin_quant/'
        else:
            if quantize_embeddings:
                ckpt_dir = './embed_quant/'
            else:
                ckpt_dir = './full_prec/'
        ckpt_dir += '{}_{}_student_{}_{}_{}'.format(bert_pre_model, student_bert_quant_type, train_state, student_quant_name, dataset_name)
        if not(quantize_bert):
            if num_bits_act != 32.0:
                print(f"Activations are being quantized to {num_bits_act} bits.")
            else:
                print(f"Activations are kept full-precision at {num_bits_act} bits.")
        if num_bits_act not in [8.0, 32.0]:
            ckpt_dir += "_quant_act"
        elif num_bits_act == 32.0:
            ckpt_dir += "_full_act"
        #ckpt_dir = data_path.joinpath(ckpt_dir)
        result_ckpt_dir = result_data_path.joinpath(ckpt_dir)
        model_ckpt_dir = model_data_path.joinpath(ckpt_dir)
        logit_ckpt_dir = logit_data_path.joinpath(ckpt_dir)
    else:
        result_ckpt_dir = checkpoint_dir
        model_ckpt_dir = checkpoint_dir
    ckpt_dir_dict = {"result_ckpt_dir": result_ckpt_dir, "model_ckpt_dir": model_ckpt_dir, "logit_ckpt_dir": logit_ckpt_dir}
    return ckpt_dir_dict

def get_pretrained_transformer_ckpt(quantize_bert, quantize_embeddings, quantize_attention, bert_pre_model, quant_name, dataset_name, bert_quant_type, 
                             num_bits_act=8.0, 
                             local_load=True, manual_load_ckpt=None, 
                             experiment_name=None, experiment_load_key=None, 
                             api_key=None, workspace=None,
                             student=False):
    if local_load:
        if manual_load_ckpt is None:
            data_path = Path(__file__).parent.parent.joinpath(f"./model_checkpoints/model_ckpts/bitbert/train")
            if student:
                print("---(Quantized) transformer is being loaded as a student model")
            if quantize_bert:
                if bert_quant_type == "PTQ":
                    print(f"A {quant_name} PFTQ BERT model quantized to bits is being used.")
                    pretrained_bert_ckpt = "./full_prec/{}_{}".format(bert_pre_model, dataset_name)
                elif bert_quant_type == "QAT":
                    print(f"A {quant_name} QAFT BERT model quantized to bits is being used.")
                    if quantize_embeddings and quantize_attention:
                        print(f"A quantized transformer with quantized embeddings is being used.")
                        pretrained_bert_ckpt = './qat/full_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
                    elif quantize_embeddings and not(quantize_attention):
                        print(f"A quantized transformer with quantized embeddings and full-prec attention is being used.")
                        pretrained_bert_ckpt = './qat/weight_emb_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
                    elif not(quantize_embeddings) and quantize_attention:
                        print(f"A quantized transformer with full precision embeddings is being used.")
                        pretrained_bert_ckpt = './qat/weight_att_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
                    else:
                        print(f"A quantized transformer with full precision embeddings and full-precision attention is being used.")
                        pretrained_bert_ckpt = './qat/lin_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
            else:
                if quantize_embeddings:
                    print(f"A full-precision transformer with quantized embeddings is being used.")
                    pretrained_bert_ckpt = './qat/embed_quant/{}_{}_{}'.format(bert_pre_model, quant_name, dataset_name)
                else:
                    print(f"A full-precision transformer is being used.")
                    pretrained_bert_ckpt = './full_prec/{}_{}'.format(bert_pre_model, dataset_name)
            if num_bits_act not in [8.0, 32.0] and bert_quant_type == "QAT":
                pretrained_bert_ckpt += "_quant_act"
            elif num_bits_act == 32.0 or bert_quant_type == "PTQ":
                pretrained_bert_ckpt += "_full_act"
            pretrained_bert_ckpt += "/checkpoint.pth"
            pretrained_bert_ckpt = data_path.joinpath(pretrained_bert_ckpt)
        else:
            pretrained_bert_ckpt = manual_load_ckpt
        
        print("pretrained_bert_ckpt", pretrained_bert_ckpt)
        ckpt = torch.load(pretrained_bert_ckpt, map_location=torch.device("cpu"))
        print("ckpt now loaded")

    else:
        from comet_ml import API
        if experiment_name is None and experiment_load_key is None:
            raise Exception("You must enter an experiment name or an experiment key from Type 3 or Type 4 models to load the BERT model.")
        load_project_name=f"bitbert_train"
        api = API(api_key=api_key)
        loaded_exp = api.get_experiment(workspace, load_project_name, experiment_load_key)
        print("experiment key being loaded: ", experiment_load_key)
        ckpt_asset = loaded_exp.get_asset_by_name("model-data/comet-torch-model.pth", return_type="binary")
        print("ckpt_asset now loaded")
        ckpt = torch.load(BytesIO(ckpt_asset), map_location="cpu")
        print("ckpt now loaded")
    return ckpt

def load_bittransgnn_for_inference(model_type, bert_pre_model,
                                  quantize_bert, quantize_embeddings, quantize_attention,
                                  bert_quant_type, 
                                  train_state, quant_name, 
                                  dataset_name, 
                                  adj_type="full",
                                  num_bits_act=8.0,
                                  local_load=None, manual_load_ckpt=None, 
                                  workspace=None, api_key=None, experiment_name=None, experiment_load_key=None,
                                  bittransgnn_inference_type="transductive"):
    if local_load:
        if manual_load_ckpt is None:
            data_path = Path(__file__).parent.parent.joinpath(f"./model_checkpoints/model_ckpts/{model_type}")
            if adj_type is not None:
                data_path = data_path.joinpath(f"./{adj_type}/train/{bittransgnn_inference_type}")
            else:
                data_path = data_path.joinpath(f"./full_adj/train/{bittransgnn_inference_type}")
            if quantize_bert:
                if quantize_embeddings and quantize_attention:
                    bittransgnn_ckpt = './full_quant/'
                elif quantize_embeddings and not(quantize_attention):
                    bittransgnn_ckpt = './weight_emb_quant/'
                elif not(quantize_embeddings) and quantize_attention:
                    bittransgnn_ckpt = './weight_att_quant/'
                else:
                    bittransgnn_ckpt = './lin_quant/'
            else:
                if quantize_embeddings:
                    bittransgnn_ckpt = './embed_quant/'
                else:
                    bittransgnn_ckpt = './full_prec/'
            bittransgnn_ckpt += '{}_{}_{}_{}_{}'.format(bert_pre_model, bert_quant_type, train_state, quant_name, dataset_name)
            #if num_bits_act not in [8.0, 32.0] and bert_quant_type == "QAT":
            #    bittransgnn_ckpt += "_quant_act"
            #elif num_bits_act == 32.0 or bert_quant_type == "PTQ":
            #    bittransgnn_ckpt += "_full_act"
            if num_bits_act not in [8.0, 32.0]:
                bittransgnn_ckpt += "_quant_act"
            elif num_bits_act == 32.0:
                bittransgnn_ckpt += "_full_act"
            bittransgnn_ckpt += "/checkpoint.pth"
            bittransgnn_ckpt = data_path.joinpath(data_path, bittransgnn_ckpt)
        else:
            bittransgnn_ckpt = manual_load_ckpt

        print("bittransgnn_ckpt", bittransgnn_ckpt)
        ckpt = torch.load(bittransgnn_ckpt, map_location="cpu")
        print("ckpt now loaded")
    else:
        from comet_ml import API
        if experiment_name is None and experiment_load_key is None:
            raise Exception("You must enter an experiment name or an experiment key from the previously trained BitTransGNN models to load the BERT model.")
        #load_project_name=f"{model_type}_{train_state}_{dataset_name}"
        load_project_name=f"bittransgnn"
        api = API(api_key=api_key)
        loaded_exp = api.get_experiment(workspace, load_project_name, experiment_load_key)
        ckpt_asset = loaded_exp.get_asset_by_name("model-data/comet-torch-model.pth", return_type="binary")
        print("ckpt_asset now loaded")
        ckpt = torch.load(BytesIO(ckpt_asset), map_location="cpu")
        print("ckpt now loaded")
    return ckpt

def get_pretrained_teacher_bittransgnn_ckpt(model_type, bert_pre_model, distillation_type,
                                     quantize_teacher_bert, quantize_teacher_embeddings, quantize_teacher_attention, 
                                     teacher_bert_quant_type, 
                                     train_state, teacher_quant_name, 
                                     dataset_name, 
                                     adj_type="full",
                                     teacher_num_bits_act=8.0,
                                     student_ckpt=None, local_load=None, manual_load_ckpt=None, 
                                     workspace=None, api_key=None, experiment_name=None, experiment_load_key=None,
                                     bittransgnn_inference_type="transductive"):
    if distillation_type == "offline":
        if local_load:
            if manual_load_ckpt is None:
                data_path = Path(__file__).parent.parent.joinpath(f"./model_checkpoints/model_ckpts/{model_type}")
                if adj_type is not None:
                    data_path = data_path.joinpath(f"./{adj_type}/train/{bittransgnn_inference_type}")
                else:
                    data_path = data_path.joinpath(f"./full_adj/train/{bittransgnn_inference_type}")
                if quantize_teacher_bert:
                    if quantize_teacher_embeddings and quantize_teacher_attention:
                        teacher_ckpt = './full_quant/'
                    elif quantize_teacher_embeddings and not(quantize_teacher_attention):
                        teacher_ckpt = './weight_emb_quant/'
                    elif not(quantize_teacher_embeddings) and quantize_teacher_attention:
                        teacher_ckpt = './weight_att_quant/'
                    else:
                        teacher_ckpt = './lin_quant/'
                else:
                    if quantize_teacher_embeddings:
                        teacher_ckpt = './embed_quant/'
                    else:
                        teacher_ckpt = './full_prec/'
                teacher_ckpt += '{}_{}_{}_{}_{}'.format(bert_pre_model, teacher_bert_quant_type, train_state, teacher_quant_name, dataset_name)
                #if teacher_num_bits_act not in [8.0, 32.0] and teacher_bert_quant_type == "QAT":
                #    teacher_ckpt += "_quant_act"
                #elif teacher_num_bits_act == 32.0 or teacher_bert_quant_type == "PTQ":
                #    teacher_ckpt += "_full_act"
                if teacher_num_bits_act not in [8.0, 32.0]:
                    teacher_ckpt += "_quant_act"
                elif teacher_num_bits_act == 32.0:
                    teacher_ckpt += "_full_act"
                teacher_ckpt += "/checkpoint.pth"
                teacher_ckpt = data_path.joinpath(data_path, teacher_ckpt)
            else:
                teacher_ckpt = manual_load_ckpt
            print("teacher_ckpt", teacher_ckpt)
            ckpt = torch.load(teacher_ckpt, map_location=torch.device("cpu"))
            print("ckpt now loaded")

        else:
            from comet_ml import API
            if experiment_name is None and experiment_load_key is None:
                raise Exception("You must enter an experiment name or an experiment key from Type 3 or Type 4 models to load the BERT model.")
            #load_project_name=f"{model_type}_{train_state}_{dataset_name}"
            load_project_name="bittransgnn"
            api = API(api_key=api_key)
            loaded_exp = api.get_experiment(workspace, load_project_name, experiment_load_key)
            print("experiment key being loaded: ", experiment_load_key)
            ckpt_asset = loaded_exp.get_asset_by_name("model-data/comet-torch-model.pth", return_type="binary")
            print("ckpt_asset now loaded")
            ckpt = torch.load(BytesIO(ckpt_asset), map_location="cpu")
            print("ckpt now loaded")
    else:
        ckpt = student_ckpt
    return ckpt

def load_bittransgnn_kd_student_for_inference(quantize_bert, quantize_embeddings, quantize_attention,
                                             bert_pre_model, student_quant_name, 
                                             dataset_name, student_bert_quant_type, 
                                             adj_type="full",
                                             num_bits_act=8.0, 
                                             local_load=True, manual_load_ckpt=None, 
                                             experiment_name=None, experiment_load_key=None, 
                                             api_key=None, workspace=None,
                                             model_type=None, train_state=None,
                                             bittransgnn_inference_type="transductive"):
    if local_load:
        if manual_load_ckpt is None:
            data_path = Path(__file__).parent.parent.joinpath(f"./model_checkpoints/model_ckpts/distill/{model_type}")
            if adj_type is not None:
                data_path = data_path.joinpath(f"./{adj_type}/train/{bittransgnn_inference_type}")
            else:
                data_path = data_path.joinpath(f"./full_adj/train/{bittransgnn_inference_type}")
            if quantize_bert:
                if quantize_embeddings and quantize_attention:
                    student_ckpt = './full_quant/'
                elif quantize_embeddings and not(quantize_attention):
                    student_ckpt = './weight_emb_quant/'
                elif not(quantize_embeddings) and quantize_attention:
                    student_ckpt = './weight_att_quant/'
                else:
                    student_ckpt = './lin_quant/'
            else:
                if quantize_embeddings:
                    student_ckpt = './embed_quant/'
                else:
                    student_ckpt = './full_prec/'
            student_ckpt += '{}_{}_student_{}_{}_{}'.format(bert_pre_model, student_bert_quant_type, train_state, student_quant_name, dataset_name)
            if num_bits_act not in [8.0, 32.0] and student_bert_quant_type == "QAT":
                student_ckpt += "_quant_act"
            elif num_bits_act == 32.0 or student_bert_quant_type == "PTQ":
                student_ckpt += "_full_act"
            student_ckpt += "/checkpoint.pth"
            student_ckpt = data_path.joinpath(data_path, student_ckpt)            
        else:
            student_ckpt = manual_load_ckpt
        
        print("student_ckpt", student_ckpt)
        ckpt = torch.load(student_ckpt, map_location="cpu")
        print("ckpt now loaded")
    else:
        from comet_ml import API
        if experiment_name is None and experiment_load_key is None:
            raise Exception("You must enter an experiment name or an experiment key from the previously trained BitTransGNN models to load the BERT model.")
        #load_project_name=f"{model_type}_kd"
        load_project_name="bittransgnn_kd"
        api = API(api_key=api_key)
        loaded_exp = api.get_experiment(workspace, load_project_name, experiment_load_key)
        ckpt_asset = loaded_exp.get_asset_by_name("model-data/comet-torch-model.pth", return_type="binary")
        print("ckpt_asset now loaded")
        ckpt = torch.load(BytesIO(ckpt_asset), map_location="cpu")
        print("ckpt now loaded")
    return ckpt
