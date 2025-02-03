from quantization.binarize_model import quantize_bitbert_for_inference
from utils import get_quant_name, load_bittransgnn_for_inference, get_model_type, get_train_state, set_bittrans_ckpt_dir
from data.loader.dataloaders import TextDataObject
from models import BitTransformer
from inference_engines import BitTransformerInference

def run_bittransgnn_for_direct_seperation(config):
        exp_configs = config["experiment_configs"]
        model_configs = config["model_configs"]
        parameters = config["parameters"]
        load_configs = config["load_configs"]
        max_length, quantize_bert, quantize_embeddings, num_bits_act, num_states = model_configs["max_length"], model_configs["quantize_bert"], \
        model_configs["quantize_embeddings"], model_configs["num_bits_act"], model_configs["num_states"]
        dataset_name, bert_pre_model, device, report_time = exp_configs["dataset_name"], exp_configs["bert_pre_model"], exp_configs["device"], exp_configs["report_time"]
        local_load, manual_load_ckpt = load_configs["local_load"], load_configs["manual_load_ckpt"]
        workspace, api_key, experiment_load_name, experiment_load_key = load_configs["workspace"], load_configs["api_key"], load_configs["experiment_load_name"], load_configs["experiment_load_key"]
        batch_size = parameters["batch_size"]
        loaded_bittransgnn_quant_type = parameters["bert_quant_type"]
        loaded_bittransgnn_joint_training = parameters["joint_training"]
        loaded_bittransgnn_quantize_gcn = model_configs["quantize_gcn"]
        checkpoint_dir = exp_configs["checkpoint_dir"]
        inference_type = exp_configs["inference_type"]
        
        train_state = get_train_state(loaded_bittransgnn_joint_training)
        model_type = get_model_type(quantize_bert, loaded_bittransgnn_quantize_gcn)
        quant_name = get_quant_name(num_states)

        text_data = TextDataObject(dataset_name, batch_size)
        nb_class = text_data.nb_class

        ckpt = load_bittransgnn_for_inference(model_type, bert_pre_model,
                                             quantize_bert, quantize_embeddings, 
                                             loaded_bittransgnn_quant_type, 
                                             train_state, quant_name, 
                                             dataset_name,
                                             local_load, manual_load_ckpt, 
                                             workspace, api_key, experiment_load_name, experiment_load_key,
                                             bittransgnn_inference_type=inference_type)
        
        ckpt_dir_dict = set_bittrans_ckpt_dir(checkpoint_dir, quantize_bert, quantize_embeddings, bert_pre_model, quant_name, dataset_name, num_bits_act, inference=True, direct_seperation=True, bittransgnn_inference_type=inference_type)

        regression = dataset_name == "stsb"

        model = BitTransformer(pretrained_model=bert_pre_model, nb_class=nb_class, quantize=quantize_bert, num_states=num_states, quantize_embeddings=quantize_embeddings, num_bits_act=num_bits_act, regression=regression)
        model = quantize_bitbert_for_inference(model, ckpt, loaded_bittransgnn_joint_training, quantize_bert, num_states, loaded_bittransgnn_quant_type, quantize_embeddings, num_bits_act)
        model = model.to(device)
        text_data.set_dataloaders_bert(model, max_length)

        inference_engine = BitTransformerInference(model, dataset_name, text_data, device)
        inference_metrics = inference_engine.run(report_time)
        return inference_metrics, ckpt_dir_dict

