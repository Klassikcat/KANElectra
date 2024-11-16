import argparse
try:
    import ujson as json
except ModuleNotFoundError:
    import json
    

def parse_SQuAD(dataset_path: str, output_path: str):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    parsed_data = []
    for dataset in data["data"]:
        for paragraph in dataset["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["is_impossible"] == False:
                    parsed_data.append(f"{qa["question"]} {qa["answers"][0]["text"]}")
                else:
                    if qa["plausible_answers"] == []:
                        parsed_data.append(qa["question"])
                        continue
                    parsed_data.append(f"{qa["question"]} {qa["plausible_answers"][0]["text"]}")
    
    with open(output_path, "w") as f:
        for line in parsed_data:
            f.write(line)
            f.write("\n")
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    parse_SQuAD(args.input_path, args.output_path)