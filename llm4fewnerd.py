import os
import jsonlines
import numpy as np
from openai import OpenAI
import argparse
import concurrent
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import ast
from pathlib import Path
import json


OPENAI_API_KEY = ""
base_url = ""

client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)

def chat(client, messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1,
        presence_penalty=1,
    )
    return response

def llm_sync_ner(data, td_dict, multi_choices=1, flag_two_stage=True, no_type_defi=False, all_types=False):
    # multi_choices: 1, 2 ，表示选择top1，top2，
    type_dict = {data["pred_label"][1]: data["pred_label"][0]}
    if multi_choices == 2:
        type_dict[data["top2"][1]] = data["top2"][0]
  
    # 得到prompt中的type
    type_list = []
    for key in type_dict:
        type_list.append(key)
    
    if all_types:
        for item in data['types']:
            if item not in type_list:
                type_list.append(item)   
    
    type_list.append("none")
    
    example = {
        "answer": "The type of the Candidate",
    }
    entity = data["entity"]
    
    if not no_type_defi:
        type_defi = ""
        for type_ in type_list:
            defi = td_dict[type_]
            type_defi += f'{type_} {defi} '
    
        question = f"Please refer to Type Definition and select the most relevant type (from Types) for Candidate in the Sentence. Answer in the format of json like:\n{example}"
    
        prompt = "Given the Type Definition, Sentence, Types, and Candidate, answer the Question base on your knowledge.\n"
        prompt += "Type Definition: " + type_defi + f"\nSentence: {data['sentence']}" + "\nTypes: " + ", ".join(type_list) + f"\nCandidate: {entity}" + f"\nQuestion: {question}"
    else:
        question = f"Please select the most relevant type (from Types) for Candidate in the Sentence. Answer in the format of json like:\n{example}"
    
        prompt = "Given the Sentence, Types, and Candidate, answer the Question base on your knowledge.\n"
        prompt += f"Sentence: {data['sentence']}" + "\nTypes: " + ", ".join(type_list) + f"\nCandidate: {entity}" + f"\nQuestion: {question}"
    
    # print(f"Prompt: {prompt}")
    
    messages = []
    answer1 = ""
    count1 = 5
    count2 = 5
    write_line = None
    while count1 > 0:
        try:
            messages.append({"role": "user", "content": prompt})
            
            response1 = chat(client, messages) # 调用api
            
            content1 = response1.choices[0].message.content
            dictionary1 = ast.literal_eval(content1)
            answer1 = dictionary1["answer"]
            
            write_line = {
                "id": data["id"],
                "entity": data["entity"],
                "logit": data["logit"],
                "pred_label": data["pred_label"],
                "gold_label": data["gold_label"],
                "top2": data["top2"],
                "is_true": data["is_true"],   
                "output": answer1,
            }
            # return write_line
            count1 = -1
            
        except:
            count1 -= 1
            print(f"Step1 id: {data['id']}, error~")
            # print("\n")
    if len(answer1) == 0:
        print(f"id {data['id']} is error~")
        return None
    if answer1 == "none" or (not flag_two_stage):
        write_line['is_entity'] = ""
        write_line["type2"] = ""
        write_line['explanation'] = ""
        return write_line
    answer2 = ""
    while count2 > 0:
        try:
            example = {
                "answer": "yes or no",
                "explanation": "Explain why the Candidate is an entity or not.",
            }
            messages.append({"role": "assistant", "content": answer1})
            prompt = f"Question: Consider the Possible Type {answer1}, whether the Candidate in the Sentence is an entity or not and explain why. "

            prompt += f"Answer in the format of json like {example}."
            
            messages.append({"role": "user", "content": prompt})
            
            response2 = chat(client, messages) # 
            
            content2 = response2.choices[0].message.content
            dictionary2 = ast.literal_eval(content2)
            
            if "answer" in dictionary2:
                answer2 = dictionary2["answer"]
                # print("Step 2 Answer: ", answer2)
                write_line["is_entity"] = answer2
            if "type" in dictionary2:
                write_line["type2"] = dictionary2["type"]
            else:
                write_line["type2"] = ""
            if "explanation" in dictionary2:
                explanation2 = dictionary2["explanation"]
                write_line["explanation"] = explanation2
                # print("Step 2 Explanation: ", explanation2)
            
            count2 = -1
        except:
            count2 -= 1
            print(f"Step2 id: {data['id']}, error~")
    if len(answer2) == 0:
        print(f"id {data['id']}'s explanation is error~")
        write_line['is_entity'] = ""
        write_line['explanation'] = ""
            
    return write_line
     
def process_ner(data, td_dict, multi_choices, prompt_version, flag_two_stage, no_type_defi, all_types):
    answer = llm_sync_ner(data, td_dict, multi_choices, prompt_version, flag_two_stage, no_type_defi, all_types)
    # print("return answer: ", answer)
    return answer

def sync_process(args, test_data, threshold, td_dict, result_path, id_set):
    # answers = []
    count = 0
    with jsonlines.open(result_path, "a") as writer:
        with ThreadPoolExecutor() as executor:
            futures = []
            for data in tqdm(test_data):
                if args.num > 0 and count >= args.num:
                    break
                if data['id'] not in id_set:
                    t1 = threshold[str(data["pred_label"][0])][0]
                    t2 = threshold[str(data["pred_label"][0])][1]
                    if data["logit"] < t1: # 筛选出需要LLM处理的
                        if data['logit'] > t2: # 只进行一阶段
                            flag_two_stage = False
                        else:
                            flag_two_stage = True
                        # print(data)
                        result = executor.submit(process_ner, data, td_dict, args.topK, flag_two_stage, args.no_td, args.all_types)
                        # print("Step1 result: ", result)
                        futures.append(result)
                        count += 1
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if not result:
                    continue
                writer.write(result)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, help="intra or inter"
    )
    parser.add_argument(
        "--N", type=int, help="N way"
    )
    parser.add_argument(
        "--K", type=int, help="K shot"
    )
    parser.add_argument(
        "--model", type=str, default="tadner", help="model name, tadner or heproto"
    )
    parser.add_argument(
        "--topK", type=int, default=2, help="use topK in multi-choices"
    )
    parser.add_argument(
        "--num", type=int, help="num of data to process, if -1, process all data"
    )
    parser.add_argument(
        "--v", type=int, help="percentile for threshold", default=50
    )
    parser.add_argument(
        "--v2", type=int, help="percentile threshold for two stage", default=50
    )
    parser.add_argument(
        "--no_td", action="store_true", help="no type definition"
    )
    parser.add_argument(
        "--all_types", action="store_true", help="use all types in prompt"
    )
    parser.add_argument(
        "--all_test_set", action="store_true", help="use all test set, not only the sampled set"
    )

    args = parser.parse_args()

    td_path = f"./type_definition.jsonl" # fewnerd的type definition

    td_dict = {}
    with jsonlines.open(td_path) as reader:
        for line in reader:
            td_dict[line["type"]] = line["definition"]

    ##############################
    if args.all_test_set:
        path = f"./preds-all-test/tadner/{args.mode}/test_{args.N}_{args.K}"
    else:
        path = f"./preds/tadner/{args.mode}/test_{args.N}_{args.K}"

    file_path = path + ".jsonl" # 读取存储好的结果

    statictics_file_path = path + "-statistics.jsonl" # 用来计算阈值

    with jsonlines.open(file_path) as reader:
        test_data = list(reader)
        
    with jsonlines.open(statictics_file_path) as reader:
        statistics = list(reader)
    statistics = statistics[0]

    threshold = {}
    v1 = args.v
    v2 = args.v2
    for key in statistics:
        threshold[key] = [np.percentile(statistics[key], v1), np.percentile(statistics[key], v2)]
        
    if args.no_td:
        extra_str = "-no_td"
    elif args.all_types:
        extra_str = "-all_types"
    else:
        extra_str = ""

    result_path = f"./llm_results/tadner/{args.mode}/test_{args.N}_{args.K}-top{args.topK}{extra_str}.jsonl"
    
    print("write to file: ", result_path)
    
    
    args.result_path = result_path
    
    args_name = f"./llm_results/tadner/{args.mode}/args/test_{args.N}_{args.K}-top{args.topK}{extra_str}.json"
    
    with Path(args_name).open("w", encoding="utf-8") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    
    id_set = []
    # 避免重复写入相同的数据
    if os.path.exists(result_path):
        with jsonlines.open(result_path, 'r') as reader:
            for line in reader:
                id_set.append(line["id"])
    
    sync_process(args, test_data, threshold, td_dict, result_path, id_set)