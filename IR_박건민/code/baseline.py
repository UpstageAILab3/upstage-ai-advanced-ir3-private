import os
import json
import time
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
# upstage embedding model

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    start_time = time.time()
    client = OpenAI(
        api_key="up_OZFOByunbTwITSBgJGCeW64CesvMo",
        base_url="https://api.upstage.ai/v1/solar"
    )
    batch_embeddings = []
    query_result = client.embeddings.create(
        model="embedding-passage",  # 인덱싱할 때는 'embedding-passage'모델로 수정해야함
        input=sentences
    )
    for query_embedding in query_result.data:
        batch_embeddings.append(query_embedding.embedding)
            
    end_time = time.time()
    print(f"get_embedding 실행시간: {end_time - start_time:.2f}초")
    return np.array(batch_embeddings).astype('float32')


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    start_time = time.time()
    query_embedding = get_embedding([query_str])[0]
    body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                    "params": {
                        "k": size,
                        "query_vector": query_embedding.tolist()
                    }
                }
            }
        }
    }
    result = es.search(index="test", body=body)
    end_time = time.time()
    print(f"dense_retrieve 실행시간: {end_time - start_time:.2f}초")
    return result


es_username = "elastic"
es_password = "OIqBKy+XsnsUC+C-*vUL"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.15.2/config/certs/http_ca.crt")

# # Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 4096,
            # brute-force 검색을 위한 인덱싱 구조 변경
            "index": False,
            
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("../data/translated_documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 역색인을 사용하는 검색 예제
search_result_retrieve = sparse_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제
search_result_retrieve = dense_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])


# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = "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"


client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
llm_model = "gpt-4o"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 인문, 사회, 과학, 공학 등 모든 분야의 지식에 통달한 전문가
## Instructions
- 사용자의 이전 메시지 정보와 주어진 참고 자료를 활용하여 간결한 답변을 생성한다.
- 주어진 검색 결과 정보로 답변할 수 없을 경우, 정보가 부족하다고 대답한다.
- 답변은 한국어로 생성한다.
"""
# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: Expert in all fields including humanities, social sciences, science, and engineering
## Instruction
- Classify the last query into question, advisory request, or casual conversation
query: Why should we document the research process and results well? // question
What is a good way to transfer an unknown powder to a scale in the lab? // advisory request
It's so hard these days.. // casual conversation
- If you have multiple conversations with the user, infer the meaning of the last conversation based on the conversation context.
- When the user asks a question or requests an explanation on various knowledge topics, the system must call the search API.
- If the user asks a question or requests an explanation on various knowledge topics, the system must call the search API.
- If the user asks a question or requests an explanation on a knowledge-related topic, the system must call the search API.
"""
# 가상문서 생성에 필요한 프롬프트
hypothetical_doc_it = "당신은 사용자의 질문 및 도움에 대해 적절한 방식으로 문서를 생성해야합니다."
# 멀티 쿼리 생성에 활용할 프롬프트
multy_query_generate = """
Generate five different versions of the given user question that aim to retrieve relevant documents from a vector database.

Use variations to capture multiple perspectives, helping to overcome the limitations of distance-based similarity searches.

- Ensure each question is distinct but related to the original question's intent.
- Focus on rewording, expanding, and adding different angles or contexts.

# Output Format

Provide five alternative questions, each on a new line. 
The output format uses only line breaks and sentences.

# Examples

input: What are the symptoms of the flu?

output:
What are the common signs and indicators of influenza?
How does flu manifest in terms of symptoms?
Could you describe the symptoms associated with the flu?
What physical symptoms might one experience with influenza?
List the typical indicators one should expect if they have the flu.
"""


# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search relevant documents. Call this whenever you got any kind of question or advisal request except casual conversation.",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query which is suitable for document search based on the user messages history"
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    
    try:
        start_time = time.time()
        result = client.chat.completions.create(
            model='gpt-4o',
            messages=msg,
            tools=tools,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
        end_time = time.time()
        print(f"LLM 호출 (질의 분석) 실행시간: {end_time - start_time:.2f}초")
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")
        
        # 가상 답변 생성
        start_time = time.time()
        multy_query = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{"role": "system", "content": multy_query_generate}] + messages,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        end_time = time.time()
        print(f"LLM 호출 (가상 답변 생성) 실행시간: {end_time - start_time:.2f}초")

        search_result = dense_retrieve(multy_query.choices[0].message.content, 3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        
        start_time = time.time()
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
            end_time = time.time()
            print(f"LLM 호출 (QA) 실행시간: {end_time - start_time:.2f}초")
        except Exception as e:
            traceback.print_exc()
            return response
        
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        start_time = time.time()
        response["answer"] = result.choices[0].message.content
        end_time = time.time()
        print(f"LLM 호출 (바로 답변) 실행시간: {end_time - start_time:.2f}초")

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            # if idx>10:
            #     break
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("../data/translated_eval.jsonl", "multyquery_output.csv")

