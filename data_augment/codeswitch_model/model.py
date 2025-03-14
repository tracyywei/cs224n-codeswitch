import argparse
import torch # type: ignore
import datasets # type: ignore

from transformers import MT5ForConditionalGeneration, MT5Tokenizer # type: ignore
from trainer import Trainer, TrainerConfig
from parsed_dataset import ParsedDataset

from codeswitch_dataset import CodeswitchDataset
from torch.utils.tensorboard import SummaryWriter               # type: ignore

def finetune_mT5_codeswitched():
    '''
    STEP 1: Finetune on codeswitched data
    '''
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    dataset = CodeswitchDataset(tokenizer=tokenizer, block_size=128)
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    # device = torch.device("cuda")
    # print(f"Device name: {torch.cuda.get_device_name(device)}")
    # total_memory = torch.cuda.get_device_properties(device).total_memory
    # print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    # allocated_memory = torch.cuda.memory_allocated(device)
    # print(f"Allocated memory: {allocated_memory / (1024**3):.2f} GB")
    # free_memory = total_memory - allocated_memory
    # print(f"Free memory: {free_memory / (1024**3):.2f} GB")

    tconf = TrainerConfig(
        max_epochs = 100,         # goal range is 5-10 epochs
        batch_size = 8,          # goal range is 8-32
        learning_rate = 2e-4,    
        betas = (0.9, 0.999), 
        weight_decay = 0.01,     # avoid overregularization
        lr_decay = True,
        warmup_tokens = 1e6,
        final_tokens = 10e9,
        num_workers = 4, 
        ckpt_path="mt5_intermediate_finetuned_ckpt.pth"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        dev_dataset=None,
        config=tconf,
        stop_early=True
    )

    trainer.train()
    torch.save(model.state_dict(), 'mt5_intermediate_finetuned_5.pth')
    

def finetune_mT5_codeswitched_generation(dataset, label_dataset):
    '''
    STEP 2: Finetune for codeswitch generation on the parsed dataset
    '''
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.load_state_dict(torch.load('mt5_intermediate_finetuned_5.pth'))   # uncomment for gpu
    # model.load_state_dict(torch.load('mt5_intermediate_finetuned.pth', map_location=torch.device('cpu')))   # uncomment for cpu
  
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    dataset = ParsedDataset(dataset, label_dataset, tokenizer=tokenizer)
    dev_dataset = ParsedDataset(dataset, label_dataset, tokenizer=tokenizer, validation=True)

    tconf = TrainerConfig(
        max_epochs=150,    
        batch_size=8,         # 16 gives OOM error
        learning_rate=4e-5,
        lr_decay=True,
        betas = (0.9, 0.98),           
        weight_decay = 0.01,   # Regularization to prevent overfitting
        num_workers=4,      # change to 4 when running on gpu (0 for cpu)
        ckpt_path="mt5_finetuned_ckpt.pth"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        dev_dataset=dev_dataset,
        config=tconf,
        stop_early=True
    )

    trainer.train()
    torch.save(model.state_dict(), 'mt5_finetuned_5.pth')


def generate_codeswitched_text(model, tokenizer, text):
    '''
    Generate codeswitched text
    '''
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_codeswitched_text_from_file(model, tokenizer, filename, output_filename):
    '''
    Generate codeswitched text from file
    '''

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
             
            # for annotated dataset
            # sentence, pos_tags, dep_rels = line.split('\t')
            # line = sentence + ' <POS> ' + pos_tags + ' <DEP> ' + dep_rels

            with open(output_filename, 'w') as out:
                codeswitched_line = generate_codeswitched_text(model, tokenizer, line)
                out.write(codeswitched_line + '\n')


def generate_codeswitched_text_batch(model, tokenizer, texts, max_length=128):
    '''
    Generate codeswitched text for a batch of inputs to speed up runtime
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        with torch.autocast("cuda"):  
            outputs = model.generate(**inputs, max_length=max_length)

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts


def generate_codewitched_text_from_dataset(model, tokenizer, output_filename, batch_size=64):
    '''
    Generate codeswitched text from dataset
    '''
    dataset = datasets.load_dataset("facebook/xnli", "en", split="train")
    reduced_length = int(len(dataset) * 0.2)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(reduced_length))

    print('Dataset loaded. Generating codeswitched text...')

    with open(output_filename, 'w') as out:
        with open('dataset/groundtruth/randomized_reduced_xnli.txt', 'w') as label_out:
            batch_premises, batch_hypotheses = [], []

            for idx, datapoint in enumerate(dataset):
                batch_premises.append(datapoint['premise'])
                batch_hypotheses.append(datapoint['hypothesis'])
                label_out.write(datapoint['premise'] + '\n')
                label_out.write(datapoint['hypothesis'] + '\n')

                # Process in batches
                if len(batch_premises) == batch_size:
                    codeswitched_premises = generate_codeswitched_text_batch(model, tokenizer, batch_premises)
                    codeswitched_hypotheses = generate_codeswitched_text_batch(model, tokenizer, batch_hypotheses)

                    for cs_premise, cs_hypothesis in zip(codeswitched_premises, codeswitched_hypotheses):
                        out.write(cs_premise + '\n')
                        out.write(cs_hypothesis + '\n')

                    batch_premises, batch_hypotheses = [], []

                if idx % (batch_size * 100) == 0:
                    print(f"Processed {idx} examples")

            # Process remaining examples (if batch size doesn't divide dataset size)
            if batch_premises:
                codeswitched_premises = generate_codeswitched_text_batch(model, tokenizer, batch_premises)
                codeswitched_hypotheses = generate_codeswitched_text_batch(model, tokenizer, batch_hypotheses)
                for cs_premise, cs_hypothesis in zip(codeswitched_premises, codeswitched_hypotheses):
                    out.write(cs_premise + '\n')
                    out.write(cs_hypothesis + '\n')


def generate_codeswitched_corpus():
    '''
    STEP 3: Generate codeswitched corpus
    '''
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.load_state_dict(torch.load('mt5_finetuned_5.pth'))   # uncomment for gpu
    # model.load_state_dict(torch.load('mt5_finetuned.pth', map_location=torch.device('cpu')))   # uncomment for cpu
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    
    # generate_codeswitched_text_from_file(model, tokenizer, "dataset/enghinglish/test.txt", "outputs/codeswitched_hinglish_en_test-3.txt")
    generate_codewitched_text_from_dataset(model, tokenizer, "outputs/codeswitched_eval.txt")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--step", type=str, help="step of ", required=True)
    args = argparser.parse_args()

    if args.step == "1":
        finetune_mT5_codeswitched()
    elif args.step == "2":
        dataset = "dataset/annotated/annotated_hinglish_en.txt"
        label_dataset = "dataset/enghinglish/dev.txt"
        finetune_mT5_codeswitched_generation(dataset, label_dataset)
    elif args.step == "3":
        generate_codeswitched_corpus()
    else:
        print("Invalid step")


if __name__ == '__main__':
    main()