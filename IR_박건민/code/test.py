import numpy as np
from openai import OpenAI
import json
import os
import ast


os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()

sys_instruction = """
Evaluate the relevance of each of the three 'reference' documents provided to answer the 'query'. Read the query and each reference document carefully to ensure you understand the query's intent and how well each document addresses it. Rate the relevance of each document using an integer score between 1 and 5, where 1 indicates not relevant at all and 5 indicates highly relevant.

# Steps

1. **Read the Query:** Understand the intent and key details of the query.
2. **Evaluate Each Reference:**
   - For each reference document, read and comprehend its content.
   - Assess how well it addresses the query's intent and whether it contains relevant information.
3. **Score Assignment:** Assign a relevance score from 1 to 5 to each reference document.

# Output Format

Provide the evaluation scores as a list of integers, each ranging from 1 to 5.

# Example

**Input:**
- Query: "[Your Query Here]"
- Reference 1: 'Contents of Reference 1'
- Reference 2: 'Contents of Reference 2'
- Reference 3: 'Contents of Reference 3'

**Output:** [3, 4, 2] (Example: the numbers are placeholders for illustration. They should be replaced with actual scored based on the task reasoning.)

# Notes

- Consider not only the presence of keywords but also the substance, context, and depth of information in each reference relative to the query.
- The evaluation should be objective, aiming to score each document fairly based on its content's relevance to the query.
- If query is empty, provide empty list

"""

msg = [{"role": "system", "content": sys_instruction}]
with open("./sample_submission.csv") as f, open("./sample_submission_eval.jsonl", "w") as of:
    docs = [json.loads(line) for line in f]
    idx = 0
    eval_score_lst = []
    for line in docs:
        contents = [reference['content'] for reference in line['references']]
        content = '-Query: '+ line['standalone_query']
        for i in range(len(contents)):
            content = content + '\n' + f'-Reference{i+1}: ' + contents[i]
        message = [{"role": "system", "content": content}] 
        result = client.chat.completions.create(
            model='gpt-4o',
            messages=msg + message,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
        print(result.choices[0].message.content)
        
        
        eval_score = result.choices[0].message.content
        if eval_score != '[]':
            eval_score_lst.append(np.array(ast.literal_eval(eval_score)).sum())
        # 평가 파일 저장
        output = {"eval_score": eval_score, "standalone_query": line['standalone_query'], "reference_doc": contents}
        of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
        idx += 1
    # 평가점수 출력
    print((np.array(eval_score_lst).mean())/15)