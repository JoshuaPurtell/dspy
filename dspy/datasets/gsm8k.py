import tqdm
import random

from datasets import load_dataset
from dspy.datasets.dataset import Dataset
import openai

class GSM8K:
    def __init__(self) -> None:
        super().__init__()
        self.do_shuffle = False

        dataset = load_dataset("gsm8k", 'main')

        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example['question']

            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))

            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']

            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))

            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]

        import dspy

        trainset = [dspy.Example(**x).with_inputs('question') for x in trainset]
        devset = [dspy.Example(**x).with_inputs('question') for x in devset]
        testset = [dspy.Example(**x).with_inputs('question') for x in testset]

        # print(f"Trainset size: {len(trainset)}")
        # print(f"Devset size: {len(devset)}")
        # print(f"Testset size: {len(testset)}")

        self.train = trainset
        self.dev = devset
        self.test = testset



def parse_integer_answer(answer, only_first_line=True):
    try:
        #if only_first_line:
            #answer = answer.strip().split('\n')[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0
    return answer


def llm_check(gold, pred, trace=None):
    llm_answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You will be given the correct answer to a math question and a student's answer. Please indicate whether the student's answer is correct or incorrect. Respond with T for correct and F for incorrect. No other response is valid."
            },
            {
                "role": "user",
                "content": f"""
# Correct answer
{gold} 

# Student answer
{pred}
"""
            }
        ],
        temperature=0.0,
        max_tokens=10,
        )
    T_or_F = llm_answer.choices[0]['message']['content']
    return T_or_F == "T"

def gsm8k_metric(gold, pred, trace=None):

    obviously_correct = int(parse_integer_answer(str(gold.answer),only_first_line=False)) == int(parse_integer_answer(str(pred.answer),only_first_line=False))
    correct = obviously_correct
    if not obviously_correct and (str(gold.answer) in str(pred.answer)):
        correct = llm_check(gold.answer, pred.answer, trace=trace)
    return correct
