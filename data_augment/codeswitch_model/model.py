import argparse
import torch # type: ignore

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

    device = torch.device("cuda")
    # print(f"Device name: {torch.cuda.get_device_name(device)}")
    # total_memory = torch.cuda.get_device_properties(device).total_memory
    # print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    # allocated_memory = torch.cuda.memory_allocated(device)
    # print(f"Allocated memory: {allocated_memory / (1024**3):.2f} GB")
    # free_memory = total_memory - allocated_memory
    # print(f"Free memory: {free_memory / (1024**3):.2f} GB")

    tconf = TrainerConfig(
        max_epochs = 10,         # goal range is 5-10 epochs
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
    torch.save(model.state_dict(), 'mt5_intermediate_finetuned_4.pth')
    

def finetune_mT5_codeswitched_generation(dataset, label_dataset):
    '''
    STEP 2: Finetune for codeswitch generation on the parsed dataset
    '''
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.load_state_dict(torch.load('mt5_intermediate_finetuned_4.pth'))   # uncomment for gpu
    # model.load_state_dict(torch.load('mt5_intermediate_finetuned.pth', map_location=torch.device('cpu')))   # uncomment for cpu
  
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    dataset = ParsedDataset(dataset, label_dataset, tokenizer=tokenizer)

    tconf = TrainerConfig(
        max_epochs=150,    
        batch_size=8,         # 16 gives OOM error
        learning_rate=4e-5,
        lr_decay=True,
        betas = (0.9, 0.98),           
        weight_decay = 0.01,   # Regularization to prevent overfitting
        num_workers=4,      # change to 2 when running on gpu (0 for cpu)
        ckpt_path="mt5_finetuned_ckpt.pth"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        dev_dataset=None,
        config=tconf,
        stop_early=True
    )

    trainer.train()
    torch.save(model.state_dict(), 'mt5_finetuned_4.pth')


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

            with open(output_filename, 'a') as out:
                codeswitched_line = generate_codeswitched_text(model, tokenizer, line)
                out.write(codeswitched_line + '\n')
                pass  


def generate_codeswitched_corpus():
    '''
    STEP 3: Generate codeswitched corpus
    '''
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.load_state_dict(torch.load('mt5_finetuned_4.pth'))   # uncomment for gpu
    # model.load_state_dict(torch.load('mt5_finetuned.pth', map_location=torch.device('cpu')))   # uncomment for cpu
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    generate_codeswitched_text_from_file(model, tokenizer, "dataset/enghinglish/test.txt", "outputs/codeswitched_hinglish_en_test-3.txt")


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