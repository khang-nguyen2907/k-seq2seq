import argparse
import json
import os
import random
import sys
import time
import traceback
from distutils.log import info
from sched import scheduler
from statistics import mode
from turtle import forward

import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from config import *
from constant_bart import *
from knowledge import KnowledgeGraph
from model_saver import *
from seed import *

from model.model_bart import *
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler

def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/KDRB_GPT2_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/vocab.json", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_train.tsv",type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_val.tsv",type=str,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_test.tsv",type=str,
                        help="Path of the testset.")
    parser.add_argument("--log_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/logs",type=str,
                        help="Path of the testset.")
    parser.add_argument("--last_logging", default=None,type=str,
                        help="Path of the testset.")

    # Model options.
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
                        help="The name of a pretrained model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size.")
    parser.add_argument("--seq_length_encoder", type=int, default=512,
                        help="Sequence length of encoder.")
    parser.add_argument("--seq_length_decoder", type=int, default=512,
                        help="Sequence length of decoder.")
    parser.add_argument("--max_length", type=int, default = 256, 
                        help= "max length.")
    parser.add_argument("--min_length", type = int, default=50, 
                        help="Min length.")
    parser.add_argument("--type_vocab_size", type = int, default=3, 
                        help="type_vocab_size of encoder config.")    

    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Warm up value.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Warm up value.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Warm up value.")
    parser.add_argument("--lr_scheduler_type", type=str, default="SchedulerType.LINEAR",
                        help="Warm up value.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Warm up value.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warm up value.")
    parser.add_argument("--num_training_steps", type=int, default=1000,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_path", default="/content/k-distilroberta-gpt2/brain/kgs/Medical.spo",type=str, help="KG name or path")

    args = parser.parse_args()
    return args

class Medical_Datset(Dataset): 
    def __init__(self, dataset_path, tokenizer,args, qna = False): 
        self.args = args
        self.dataset_path = dataset_path
        self.knowledge = KnowledgeGraph(txt_path=args.kg_path)
        self.vocab_file = self.load_vocab() 
        self.sentences = self.load_sentences()
        self.columns = self.load_columns()
        self.tokenizer = tokenizer
        self.qna = qna
        
    def load_sentences(self): 
        sentences = []
        with open(self.dataset_path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        return sentences
    
    def load_columns(self): 
        columns = {}
        with open(self.dataset_path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                try:
                    line = line.strip().split("\t")
                    if line_id == 0:
                        for i, column_name in enumerate(line):
                            columns[column_name] = i
                        continue
                except:
                    pass
        return columns

    def load_vocab(self):
        f = open(self.args.vocab_path)
        vocab = json.load(f)
        f.close()
        print(200 * '-')
        print("Vocabulary Size: ", len(vocab))
        print(200 * '-')
        return vocab
    
    def __len__(self): 
        return len(self.sentences)

    def __getitem__(self, idx): 
        line = self.sentences[idx]
        line = line.strip().split('\t')
        if len(line) == 2 and self.qna == False: 
            label = str(line[self.columns['answer']])
            text = str(line[self.columns['question']])
            tokens, pos, vm = self.knowledge.add_knowledge_with_vm([text], max_length = self.args.seq_length)

            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")
            token_ids = [self.vocab_file.get(t) for t in tokens]
            mask = [1 if t != PAD_TOKEN else 0 for t in tokens]
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(label, padding = "max_length", truncation = True, max_length = self.args.seq_length).input_ids

            input_ids = torch.LongTensor(token_ids)   
            visible_matrix = torch.LongTensor(vm)
            label_ids = torch.LongTensor(labels)
            position_ids = torch.LongTensor(pos)
            mask_ids = torch.LongTensor(mask)

            return input_ids, visible_matrix, label_ids, position_ids, mask_ids
               
def main():
    #####################################################################################################
    #ARGS
    args = parsers()
    args = load_hyperparam(args)
    set_seed(args.seed)
    #####################################################################################################
    #MODEL 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = True)
    config = AutoConfig.from_pretrained(args.model_name)
    num_return = 4
    gen_kwargs = {
        "do_sample": True,
        "early_stopping": True, 
        "num_beams": num_return, 
        "length_penalty": .9, 
        "num_return_sequences": num_return,
        "max_length": args.max_length, 
        "min_length": args.min_length, 
        "top_k": 50, 
        "top_p": 0.95
    }
    for k, v in gen_kwargs.items(): 
        setattr(config, k, v)
    model = BartModel.from_pretrained(
        args.model_name, 
        config = config
    ).to(device)

    if args.pretrained_model_path != "None": 
        print(200*"-")
        print("CONTINUE TO TRAIN THE MODEL FROM: {}".format(args.pretrained_model_path))
        print(200*'-')
        model.load_state_dict(torch.load(args.pretrained_model_path), strict = False)
    else: 
        print(200*"-")
        print("THE MODEL IS INITIALIZED FROM SCRATCH")
        print(200*'-')
    start_epoch = 1
    last_epoch = 0
    best_result = 9999.0

    print("Best result before training: ", best_result)
    if args.last_logging != "None": 
        print(200*"-")
        print("LOADING LOGGING INFORMATION FROM {}".format(args.last_logging))
        last_logger = open(args.last_logging)
        logger_info = json.load(last_logger)
        last_epoch = logger_info['epoch']
        print("Previous epoch: ", last_epoch)
        last_loss = logger_info['total_loss']
        print("Previous loss: ", last_loss)
        best_result = last_loss
        start_epoch += last_epoch
        print("Previous best result: ", best_result)
        print("start_epoch: {0} || last_epoch: {1}".format(start_epoch, last_epoch + args.epochs_num + 1))
        print(200*'-')
    
    #####################################################################################################
    #EVALUATION 
    rouge = datasets.load_metric("rouge")

    def evaluate(args, is_test):
        if is_test: 
            eval_dataset = Medical_Datset(args.test_path, tokenizer, args)
            eval_loader = DataLoader(dataset = eval_dataset, batch_size=args.batch_size)
        else: 
            eval_dataset = Medical_Datset(args.dev_path, tokenizer, args)
            eval_loader = DataLoader(dataset = eval_dataset, batch_size = args.batch_size)
        
        model.eval()
        eval_loop  = tqdm(enumerate(eval_loader), total = len(eval_loader))
        for eval_batch_idx, eval_inputs in eval_loop: 
            input_ids = eval_inputs[0].to(device)
            visible_matrix = eval_inputs[1].to(device)
            label_ids = eval_inputs[2].to(device)
            position_ids = eval_inputs[3].to(device)
            mask_ids = eval_inputs[4].to(device)

            pred_ids = model.generate(
                input_ids   
            )
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            rouge1_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge1"])["rouge1"].mid
            rouge2_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge2"])["rouge2"].mid
            rougeL_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rougeL"])["rougeL"].mid


            rouge1_precision = round(rouge1_output.precision, 4) 
            rouge1_recall = round(rouge1_output.recall, 4)
            rouge1_fmeasure = round(rouge1_output.fmeasure, 4)
            
            rouge2_precision = round(rouge2_output.precision, 4) 
            rouge2_recall = round(rouge2_output.recall, 4)
            rouge2_fmeasure = round(rouge2_output.fmeasure, 4)
            
            rougeL_precision = round(rougeL_output.precision, 4) 
            rougeL_recall = round(rougeL_output.recall, 4)
            rougeL_fmeasure = round(rougeL_output.fmeasure, 4)
            metrics_result = {
                "rouge1_precision" : rouge1_precision, 
                "rouge1_recall": rouge1_recall, 
                "rouge1_fmeasure": rouge1_fmeasure, 
                "rouge2_precision": rouge2_precision, 
                "rouge2_recall": rouge2_recall, 
                "rouge2_fmeasure": rouge2_fmeasure, 
                "rougeL_precision": rougeL_precision, 
                "rougeL_recall": rougeL_recall, 
                "rougeL_fmeasure": rougeL_fmeasure
            }
            if is_test: 
                print("report TEST: rouge1_precision: {0} \trouge1_recall: {1} \trouge1_fmeasure: {2}".format(rouge1_precision, rouge1_recall, rouge1_fmeasure))
                print("report TEST: rouge2_precision: {0} \trouge2_recall: {1} \trouge2_fmeasure: {2}".format(rouge2_precision, rouge2_recall, rouge2_fmeasure))
                print("report TEST: rougeL_precision: {0} \trougeL_recall: {1} \trougeL_fmeasure: {2}".format(rougeL_precision, rougeL_recall, rougeL_fmeasure))
                return metrics_result
            else: 
                print("report VAL: rouge1_precision: {0} \trouge1_recall: {1} \trouge1_fmeasure: {2}".format(rouge1_precision, rouge1_recall, rouge1_fmeasure))
                print("report VAL: rouge2_precision: {0} \trouge2_recall: {1} \trouge2_fmeasure: {2}".format(rouge2_precision, rouge2_recall, rouge2_fmeasure))
                print("report VAL: rougeL_precision: {0} \trougeL_recall: {1} \trougeL_fmeasure: {2}".format(rougeL_precision, rougeL_recall, rougeL_fmeasure))
                return metrics_result

    #####################################################################################################
    #TRAINING PHASE 
    print("************************Start training************************")
    train_dataset = Medical_Datset(args.train_path, tokenizer, args)
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    
    instances_num = len(train_dataset)
    num_training_steps = args.epochs_num * len(train_loader)
    args.report_steps = len(train_loader)

    print("Batch size: ", args.batch_size)
    print("The number of training instances:", instances_num)

    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear", 
        optimizer = optimizer, 
        num_warmup = 0, 
        num_training_steps = num_training_steps
    )

    #Training Network 
    total_loss = 0.
    result = 0.0 
    for epoch in range(start_epoch, last_epoch + args.epochs_num + 1):
        print("\n")
        print(150 * '-')
        t1 = time.time()
        info = {}
        total_losses = []
        losses = []
        model.train()
        loop = tqdm(enumerate(train_loader), total = len(train_loader))
        for batch_idx, inputs in loop: 
            model.zero_grad()

            input_ids = inputs[0].to(device)
            visible_matrix = inputs[1].to(device)
            label_ids = inputs[2].to(device)
            position_ids = inputs[3].to(device)
            mask_ids = inputs[4].to(device)

            outputs = model(
                input_ids = input_ids, 
                attention_mask = visible_matrix, 
                position_ids = position_ids, 
                labels = label_ids
            )
            loss = outputs[0]

            losses.append(loss.item())
            total_loss += loss.item()
            
            #backward 
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            #update progress bar 
            loop.set_description(f"Epoch [{epoch/args.epochs_num}]")
            loop.set_postfix(loss = loss.item())
            
        print("Avg. loss: ", total_loss/args.report_steps)
        total_loss = 0.

        print("\n***********Start evaluation on dev dataset***********")
        result = evaluate(args, False)
        

        print("\n***********Start evaluation on test dataset***********")
        rt = evaluate(args, True)
        
        t2 = time.time()
        info['epoch'] = int(epoch)
        info['total_loss'] = float(total_losses[-1])
        info['loss'] = losses
        info['val'] = result
        info['test'] = rt
        info['time'] = t2-t1
        path_log = os.path.join(args.log_path, "log_epoch_"+str(epoch)+".json")
        with open(path_log, mode = "w") as outfile: 
            json.dump(info, outfile)

        ttl = float(total_losses[-1])
        if ttl < best_result: 
            best_result = ttl
            save_model(model, args.output_model_path)

    #Evaluation phase 
    print("\nFinal evaluation on the test dataset")

    if torch.cuda.device_count()>1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else: 
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)
    print("\nTraining progress completed.")

if __name__ == "__main__": 
    main()