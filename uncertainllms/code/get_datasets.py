import datasets

# load datasets from huggingface


# main
if __name__ == '__main__':
    # get datasets
    squad2 = datasets.load_dataset("squad_v2")
    squad2.save_to_disk("../UncertainLLMs/data/squad_v2")
