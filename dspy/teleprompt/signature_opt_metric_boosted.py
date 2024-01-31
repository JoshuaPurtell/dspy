import inspect
import re
from collections import defaultdict
from typing import List

import pandas as pd
import tiktoken

import dsp
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter, and evaluate it on an end task:

teleprompter = SignatureOptimizer(prompt_model=prompt_model, metric=metric, breadth=BREADTH, depth=DEPTH, init_temperature=INIT_TEMPERATURE)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program.deepcopy(), devset=devset[:DEV_NUM], eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* breadth: The number of new prompts to generate at each iteration. Default=10.
* depth: The number of times we should ask our prompt model to genereate new prompts, with the history of the past prompts as input. Default=3.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.4.
* verbose: Tells the method whether or not to print intermediate steps.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track the following statistics:
                    * results_best: The min,max,avg,stddev of top 10 scores for each predictor at each depth.
                    * results_latest: The min,max,avg,stddev of newest prompt scores for each predictor at each depth.
                    * total_calls: The total number of calls to the task metric.
                These statistics will be returned as attributes of the best program.
"""
class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class GenerateInstructionGivenAttempts(dspy.Signature):
        """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

        attempted_instructions = dspy.InputField(format=dsp.passages2text)
        current_metric_observations_data_boosted = dspy.InputField(format=dsp.passages2text, desc="Summaries for metric_observations generated recently, with exemplars of where they failed included")
        historical_metric_observation_summaries = dspy.InputField(format=dsp.passages2text, desc="Summaries for the historical metric_observations")
        proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class ProposeMetricObservation(dspy.Signature):
    """# Premise
You are an expert programmer tasked with identifying systematic errors or weaknesses in an AI system that generates answers to posed questions. You are provided with the following information:
* A list summaries of previously generated metric_observations that capture past feedback on avoiding systematic errors or weaknesses in the system.
* A python function that scores the AI system's output as correct or incorrect. This is the metric that the system is trying to optimize - use it as your guide.
* The current prompt used by the system.
* A list of previously tried prompts, along with their scores on a dataset of sample questions. The score is the percentage of questions answered correctly.
* A sample of question / answer pairs generated by the system that are deemed correct. Indices will be indicated with 'Question at Index i' (in this case for index i).
* A sample of question / answer pairs generated by the system that are deemed incorrect.

We help the AI system to avoid errors by providing it metric_observations. metric_observations have the following attributes:
* Summary: a one-sentence summary of the problem that the tip is meant to address.
* Suggestion: a one-sentence suggestion for how to address the problem
* Indices: a list of at most 3 indices of the failed examples that the tip is meant to address.

Some notes regarding metric_observations:
* metric_observations are meant to be generalizable to future examples, and not just the examples that they are associated with. They exist to help the system avoid systematic errors or weaknesses.
* The 'suggestion' component of a tip is meant to be actionable. It should address a specific improvement that the system can make. Do not omit this section or make it vague.
* metric_observations MUST NOT reference arbitrary details specific to the examples they are associated with. Such metric_observations would actively degrade system performance.
* The list of metric_observations should be considered as a whole. New metric_observations which rehearse the same issues as existing metric_observations should not be added, neither new metric_observations that seem much less impactful than existing metric_observations.
* metric_observations should address process, not specific details. For instance, a tip should not have the summary 'the AI system sometimes calculates net profit incorrectly'. Instead, it should have the summary 'the AI system sometimes fails to carefully convert its answer to the final format'. The former is a specific detail, while the latter is a process that can be applied to many examples.

# Objective
Your objective is to identify systematic errors or weaknesses in the system that are not captured by the already-existing metric_observations.
Refer to the python function that scores the system's generation as correct or incorrect in order to identify why failed examples are likely failing, and to infer what changes could lead to successful generations.
Although you should strive to increase overall performance, keep in mind that the AI's success is purely captured by the provided metric. Strive to increase the metric.
Do so by identifying problems at medium levels of abstraction that help explain multiple failed examples.
Focus on identify root causes of errors, rather than symptoms of errors. If one error causes others downstream, focus on the root cause.

# Constraints
* Do not return an empty list of metric_observations unless there are no improvements to be made.
* Provide no more than 2 metric_observations. Ensure you provide only the most impactful metric_observations.
* Do not duplicate existing metric_observations or provide duplicate metric_observations.
* Any given example index can be associated with at most one tip. If an example has multiple issues, choose the most impactful one. If the same index is associated with multiple metric_observations, your output will be deemed invalid.
* For a given tip, only include at most 3 exemplars. If a tip has more than 3 exemplars, choose the ones that most clearly demonstrate the principle.
* Each new tip will have the following information associated with it: a summary of the tip, and a list of indices of the failed examples that the tip is meant to address.
* Return your new metric_observations in this form: a summary and a list of indices.
* Ensure metric_observations are impactful - try not to make trivial suggestions that are unlikely to substantially improve performance.
* Ensure that you don't cover the same ground as the existing metric_observations - you must identify new systematic errors or weaknesses. If the old metric_observations cover current issues, leave them as they are and do not rehearse them.
* Strive for perfect precision, clarity, and concreteness
* Return nothing other than the metric_observations. Do not return any other information.
* Never directly reference the scoring function - translate any important information you want to convey into an operationalizable tip.

# Example metric_observations

## Tip 1
* Summary: The system sometimes correctly computes an intermediate result, but stops before moving onto the final answer.
* Suggestion: Sketch a brief outline of the that need to be taken, so that no steps are missed.
* Indices: [0, 19]

## Tip 2
* Summary: The system sometimes uses incorrect units.
* Suggestion: Check that the units are correct.
* Indices: [9, 10]
"""
    current_instructions = dspy.InputField(format=dsp.passages2text, desc="The current instructions")
    sample_questions_the_current_instructions_fail_on = dspy.InputField(format=dsp.passages2text, desc="The sample questions the current instructions fail on")
    sample_questions_the_current_instructions_succeed_on = dspy.InputField(format=dsp.passages2text, desc="The sample questions the current instructions succeed on")
    historical_instructions = dspy.InputField(format=dsp.passages2text, desc="The historical instructions")
    historical_metric_observations = dspy.InputField(format=dsp.passages2text, desc="The historical metric_observations")
    ai_metric = dspy.InputField(format=dsp.passages2text, desc="The metric the AI is trying to optimize")
    proposed_metric_observation_summary = dspy.OutputField(desc="A summary for the proposed tip")
    indices_in_provided_failed_examples_that_exemplify_proposed_metric_observation = dspy.OutputField(desc="Indices for failed exemplars for the proposed tip. Return a stringified list.")# Must be a stringified list of integers in the form '[i1, i2, i3]' where i1, i2, and i3 are integer indices.

class ReconcileMetric_Observations(dspy.Signature):
    """# Premise
You are an expert programmer tasked with consolidating a long list of metric_observations that have been submitted by other experts in order to help an AI system avoid systematic errors or weaknesses in its answers.
The AI system in question has improved over time, so it is possible that some of the metric_observations are no longer relevant.
Moreover, although the experts were asked to not submit duplicate or mostly-overlapping metric_observations, it is possible that some of the metric_observations are redundant.
You are provided with the following information:
* Each of the current metric_observations, which have the following attributes:
    * Summary: a one-sentence summary of the problem that the tip is meant to address.
    * Suggestion: a one-sentence suggestion for how to address the problem
    * Indices: a list of indices of the failed examples that the tip is meant to address.

# Objective
Your objective is to consolidate the metric_observations into a smaller list of metric_observations that are more impactful and less redundant. Your list should be no longer than 6 metric_observations.
You will do this using the following process:
1. Identify metric_observations that are likely to address problems so minor that they are unlikely to substantially improve performance. Remove these metric_observations from the list.
2. Identify metric_observations that are largely redundant with other metric_observations. For any set of redundant metric_observations, remove all but one of them.
3. Identify metric_observations that are related to each other, but not redundant. Consolidate these metric_observations into a single tip.

This will leave you with a list of valid metric_observations that are impactful and non-redundant. From this list, you will select at random ONE of the most impactful 6 metric_observations to return.

# Constraints
* You must return only 1 tip.
* Take care to return the parent TIP indices for the child tip, not the parent EXAMPLE indices.

## Formatting
Return the following information for each tip:
- Summary: a one-sentence summary of the problem that the tip is meant to address, together with a one-sentence suggestion for how to address the problem.
- Tip Parents: the indices of the metric_observations that this tip is related to. For metric_observations that result from consolidation, this will be the indices of the metric_observations that were consolidated to form this tip. For metric_observations that are not consolidated, this will be a single index."""
    historical_metric_observations = dspy.InputField(desc="The historical metric_observations")
    reconciled_metric_observation_summaries = dspy.OutputField(desc="A summary for the chosen reconciled tip")
    reconciled_metric_observation_failure_exemplars = dspy.OutputField(desc="Indices that identify where parent metric_observations of the reconciled tip are located within the provided list of historical metric_observations. Return a stringified list of integers in the form '[i1, i2, i3]' where i1, i2, and i3 are integer indices.")

def safe_strip(text: str) -> List[int]:
    matches = re.findall(r'\d+',str(text))
    indices = [int(match) for match in matches]
    return indices

def exemplars_to_df(exemplars: list):
    d = {k: [exemplars[i][k] for i in range(len(exemplars))] for k in exemplars[0].keys()}
    return pd.DataFrame(data=d)

def safely_markdownify(df: pd.DataFrame):
    try:
        markdown_str = ""
        for index, row in df.iterrows():
            markdown_str += f"EXAMPLE AT INDEX {index}\n"
            markdown_str += row.to_markdown() + "\n\n"
        return markdown_str
    except:
        return "None Shown - Too Long"

class MetricObservation:
    summary: str
    failure_exemplars: pd.DataFrame
    

    def __init__(self, summary: str, failure_exemplars: pd.DataFrame):
        self.summary = summary
        self.failure_exemplars = failure_exemplars
        self.example_token_budget = 300
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def __str__(self):
        budget_remaining = self.example_token_budget
        nrows_to_keep = 0
        while True:
            token_count = len(self.encoder.encode(' '.join(self.failure_exemplars.iloc[nrows_to_keep].astype(str))))
            if token_count > budget_remaining:
                break
            nrows_to_keep += 1
            if nrows_to_keep >= len(self.failure_exemplars) -1:
                break
            budget_remaining -= token_count
        if nrows_to_keep == 0:
            return f"Summary: {self.summary}\nFailure Exemplars: None Shown - {len(self.failure_exemplars)} present, too long to show"
        return f"Summary: {self.summary}\nFailure Exemplars: {safely_markdownify(self.failure_exemplars[0:nrows_to_keep])}"

class MetricObservationHistory:

    def __init__(self, devset: pd.DataFrame, evalset: pd.DataFrame, metric=None):
        self.max_metric_observations = 15
        self.current_generation_metric_observations: List[MetricObservation] = []
        self.previous_generations_metric_observations_used: List[MetricObservation] = []
        self.previous_generations_metric_observations_unused: List[MetricObservation] = []
        self.devset: list = devset
        self.devset_df: pd.DataFrame = exemplars_to_df(devset)
        self.evalset: list = evalset
        self.evalset_df: pd.DataFrame = exemplars_to_df(evalset)
        self.tokens_incorrect_example = 2000
        self.tokens_correct_example = 600
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.metric = metric


class SignatureOptimizerMetricBoosted(Teleprompter):
    def __init__(self, prompt_model=None, task_model=None,metric=None, breadth=10, depth=3, init_temperature=1.4, verbose=False, track_stats=False, log_dir=None):
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.task_model = task_model
        self.verbose = verbose
        self.track_stats = track_stats
        self.metric_observation_history = None

    def budgeted_df_stringify(self, df: pd.DataFrame, budget: int):
            budget_remaining = budget
            nrows_to_keep = 0
            while nrows_to_keep < len(df):
                token_count = len(self.metric_observation_history.encoder.encode(' '.join(df.iloc[nrows_to_keep].astype(str))))
                if token_count > budget_remaining:
                    break
                nrows_to_keep += 1
                budget_remaining -= token_count
            if nrows_to_keep == 0:
                return f"None Shown - {len(df)} present, too long to show"
            return safely_markdownify(df[0:nrows_to_keep])
    
    def propose_metric_observations(self, current_instructions, current_instructions_failed_on, current_instructions_succeeded_on, historical_instructions, init_temperature=0.3):
        historical_metric_observations = [metric_observation.summary for metric_observation in self.metric_observation_history.previous_generations_metric_observations_used]
        success_stringified = self.budgeted_df_stringify(current_instructions_succeeded_on, self.metric_observation_history.tokens_correct_example)
        failure_stringified = self.budgeted_df_stringify(current_instructions_failed_on, self.metric_observation_history.tokens_incorrect_example)
        with dspy.settings.context(lm=self.prompt_model):
            new_metric_observation_generations = dspy.Predict(ProposeMetricObservation, n=3, temperature=init_temperature)(
                current_instructions=current_instructions,
                sample_questions_the_current_instructions_succeed_on=success_stringified,
                sample_questions_the_current_instructions_fail_on=failure_stringified,
                historical_instructions=historical_instructions,
                ai_metric=f"```{inspect.getsource(self.metric_observation_history.metric)}```",
                historical_metric_observations=historical_metric_observations)

        if "None Shown" in failure_stringified:
            new_metric_observation_generations.completions.indices_in_provided_failed_examples_that_exemplify_proposed_metric_observation = [[] for i in range(len(new_metric_observation_generations.completions.indices_in_provided_failed_examples_that_exemplify_proposed_metric_observation))]
        new_metric_observations = []
        for i in range(len(new_metric_observation_generations.completions)):
            safe_citations = safe_strip(new_metric_observation_generations.completions.indices_in_provided_failed_examples_that_exemplify_proposed_metric_observation[i])
            if len(safe_citations) == 0:
                raise ValueError("Exemplar parsing failed")
            if len([index for index in safe_citations if index < len(self.metric_observation_history.devset)])<len(safe_citations):
                print("Warning: Exemplar parsing yielded invalid indices")
            safe_citations = [index for index in safe_citations if index < len(self.metric_observation_history.devset)]
            new_metric_observations.append(MetricObservation(new_metric_observation_generations.completions.proposed_metric_observation_summary[i], self.metric_observation_history.devset_df.loc[safe_citations]))
        self.metric_observation_history.previous_generations_metric_observations_used.extend(self.metric_observation_history.current_generation_metric_observations)
        self.metric_observation_history.current_generation_metric_observations = new_metric_observations
        print([metric_observation.summary for metric_observation in self.metric_observation_history.previous_generations_metric_observations_used])
    def update_metric_observations(self):
        if len(self.metric_observation_history.previous_generations_metric_observations_used) > self.metric_observation_history.max_metric_observations:
            with dspy.settings.context(lm=self.prompt_model):
                historical_metric_observations_with_indices = ""
                for i,metric_observation in enumerate(self.metric_observation_history.previous_generations_metric_observations_used):
                    historical_metric_observations_with_indices += f"Tip at Index {i}: {metric_observation.summary}\n"
                    historical_metric_observations_with_indices += f"Related Data Indices Where System Failed: {metric_observation.failure_exemplars.index}\n"
                reconciled_metric_observations_generation = dspy.Predict(ReconcileMetric_Observations, n=self.metric_observation_history.max_metric_observations//2, temperature=self.init_temperature)(historical_metric_observations=historical_metric_observations_with_indices)
            self.metric_observation_history.previous_generations_metric_observations_unused.extend(self.metric_observation_history.previous_generations_metric_observations_used)
            for i in range(len(reconciled_metric_observations_generation.completions)):
                safe_parent_indices = safe_strip(reconciled_metric_observations_generation.completions.reconciled_metric_observation_failure_exemplars[i])
                if len([safe_parent_index for safe_parent_index in safe_parent_indices if safe_parent_index < len(self.metric_observation_history.previous_generations_metric_observations_used)])<len(safe_parent_indices):
                    print("Warning: Exemplar parsing yielded invalid indices")
                    safe_parent_indices = [safe_parent_index for safe_parent_index in safe_parent_indices if safe_parent_index < len(self.metric_observation_history.previous_generations_metric_observations_used)]
                corresponding_parent_metric_observations = [self.metric_observation_history.previous_generations_metric_observations_used[i] for i in safe_parent_indices]
                corresponding_parent_indices = list(set([index for metric_observation in corresponding_parent_metric_observations for index in metric_observation.failure_exemplars.index]))
                self.metric_observation_history.previous_generations_metric_observations_used.append(MetricObservation(reconciled_metric_observations_generation.completions.reconciled_metric_observation_summaries[i], self.metric_observation_history.devset_df.loc[corresponding_parent_indices]))
            

    def _check_candidates_equal(self, candidate1, candidate2):
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors()):
            if not p1.extended_signature.instructions == p2.extended_signature.instructions:
                return False
            if not p1.extended_signature.fields[-1] == p2.extended_signature.fields[-1]:
                return False
        return True

    def _drop_duplicates(self, candidates):
        final_candidates = []
        last_batch = []
        last_batch_score = -1
        for c in candidates:
            repeat = False
            if c['score'] == last_batch_score:
                for c2 in last_batch:
                    if (self._check_candidates_equal(c, c2)):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(c)
            else:
                last_batch = [c]
                last_batch_score = c['score']
            if not repeat:
                final_candidates.append(c)
        return final_candidates
    
    def compile(self, student, *, devset, evalset, eval_kwargs):
        self.metric_observation_history = MetricObservationHistory(devset, evalset, metric=self.metric)
        """student is a program that needs to be optimized, note that it may be zero-shot or already pre-optimized for demos != []"""
        module = student.deepcopy()
        evaluate = Evaluate(devset=devset, metric=self.metric, **eval_kwargs)
        total_calls = 0
        results_best = {id(p):{"depth": [], "max": [], "average": [], "min":[], "std": []} for p in module.predictors()}
        results_latest = {id(p):{"depth": [], "max": [], "average": [], "min":[], "std": []} for p in module.predictors()}

        if self.track_stats:
            import numpy as np


        candidates = {}
        evaluated_candidates = defaultdict(dict)

        for predictor in module.predictors():
            basic_instruction = None
            basic_prefix = None
            if (hasattr(predictor, 'extended_signature')):
                basic_instruction = predictor.extended_signature.instructions
                basic_prefix = predictor.extended_signature.fields[-1].name
            else:
                basic_instruction = predictor.extended_signature1.instructions
                basic_prefix = predictor.extended_signature1.fields[-1].name
            if self.prompt_model: 
                with dspy.settings.context(lm=self.prompt_model):
                    instruct = dspy.Predict(BasicGenerateInstruction, n=self.breadth-1, temperature=self.init_temperature)(basic_instruction=basic_instruction)
            else:
                instruct = dspy.Predict(BasicGenerateInstruction, n=self.breadth-1, temperature=self.init_temperature)(basic_instruction=basic_instruction)
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}
        
        if self.verbose and self.prompt_model: print(f"{self.prompt_model.inspect_history(n=1)}")

        latest_candidates = candidates
        all_candidates = candidates
        
        module_clone = module.deepcopy()

        for d in range(self.depth):
            if self.verbose: print(f"Starting iteration {d}/{self.depth}.")

            latest_scores = []
        
            for p_i, (p_old, p_new) in enumerate(zip(module.predictors(), module_clone.predictors())):
                candidates_ = latest_candidates[id(p_old)]
                if len(module.predictors()) > 1:
                    candidates_ = all_candidates[id(p_old)] 

                for c_i, c in enumerate(candidates_):
                    instruction, prefix = c.proposed_instruction.strip('"').strip(), c.proposed_prefix_for_output_field.strip('"').strip()

                    if (hasattr(p_new, 'extended_signature')):
                        p_new.extended_signature.instructions = instruction
                        p_new.extended_signature.fields[-1] = p_new.extended_signature.fields[-1]._replace(name=prefix)
                    else:
                        p_new.extended_signature1.instructions = instruction
                        p_new.extended_signature1.fields[-1] = p_new.extended_signature1.fields[-1]._replace(name=prefix)
                        p_new.extended_signature2.instructions = instruction
                        p_new.extended_signature2.fields[-1] = p_new.extended_signature2.fields[-1]._replace(name=prefix)           

                    if self.verbose: print(f"----------------")
                    for i,predictor in enumerate(module_clone.predictors()):
                        if self.verbose: print(f"Predictor {i}")
                        if (hasattr(predictor, 'extended_signature')):
                            if self.verbose: print(f"i: {predictor.extended_signature.instructions}")
                            if self.verbose: print(f"p: {predictor.extended_signature.fields[-1].name}")
                        else:
                            if self.verbose: print(f"i: {predictor.extended_signature1.instructions}")
                            if self.verbose: print(f"p: {predictor.extended_signature1.fields[-1].name}")
                        if self.verbose: print()
                    if self.verbose: print(f"At Depth {d}/{self.depth}, Evaluating Prompt Candidate #{c_i}/{len(candidates_)} for Predictor {p_i} of {len(module.predictors())}.")
                    
                    score = evaluate(module_clone, devset=devset, **eval_kwargs)
                    devset_annotated = evaluate.dataset.copy()
                    devset_failed = devset_annotated[devset_annotated["correct"] == False]
                    devset_succeeded = devset_annotated[devset_annotated["correct"] == True]
                    k = 5
                    historical_instructions = sorted(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate['score'], reverse=True)[:min(k, len(evaluated_candidates[id(p_old)].values()))]
                    self.propose_metric_observations(instruction, devset_failed, devset_succeeded,historical_instructions)
                    self.update_metric_observations()
                    if self.verbose and self.prompt_model: print(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                    total_calls += 1
                    if self.verbose: print(f"----------------")

                    replace_entry = True
                    if self.verbose: print(f"(instruction, prefix) {(instruction, prefix)}")
                    if ((instruction, prefix) in evaluated_candidates[id(p_old)]):
                        if evaluated_candidates[id(p_old)][(instruction, prefix)]["score"] >= score:
                            replace_entry = False

                    if replace_entry:
                        evaluated_candidates[id(p_old)][(instruction, prefix)] = {
                            "score": score,
                            "program": module_clone.deepcopy(),
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth": d
                        }
                    
                    if (len(candidates_)-self.breadth <= c_i):
                        latest_scores.append(score)

                if self.track_stats:
                    results_latest[id(p_old)]["depth"].append(d)
                    results_latest[id(p_old)]["max"].append(max(latest_scores))
                    results_latest[id(p_old)]["average"].append(sum(latest_scores)/len(latest_scores))
                    results_latest[id(p_old)]["min"].append(min(latest_scores))
                    results_latest[id(p_old)]["std"].append(np.std(latest_scores))
                
                best_candidate = max(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate['score'])
                if (hasattr(p_new, 'extended_signature')):
                    p_new.extended_signature.instructions = best_candidate["instruction"]
                    p_new.extended_signature.fields[-1] = p_new.extended_signature.fields[-1]._replace(name=best_candidate["prefix"])
                else:
                    p_new.extended_signature1.instructions = best_candidate["instruction"]
                    p_new.extended_signature1.fields[-1] = p_new.extended_signature1.fields[-1]._replace(name=best_candidate["prefix"])
                    p_new.extended_signature2.instructions = best_candidate["instruction"]
                    p_new.extended_signature2.fields[-1] = p_new.extended_signature2.fields[-1]._replace(name=best_candidate["prefix"])     
                if self.verbose: print(f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\np: {best_candidate['prefix']}")
                if self.verbose: print(f"Full predictor with update: ")
                for i,predictor in enumerate(module_clone.predictors()):
                    if self.verbose: print(f"Predictor {i}")
                    if (hasattr(predictor, 'extended_signature')):
                        if self.verbose: print(f"i: {predictor.extended_signature.instructions}")
                        if self.verbose: print(f"p: {predictor.extended_signature.fields[-1].name}")
                    else:
                        if self.verbose: print(f"i: {predictor.extended_signature1.instructions}")
                        if self.verbose: print(f"p: {predictor.extended_signature1.fields[-1].name}")
                    if self.verbose: print()

            if d == self.depth-1:
                break

            
            new_candidates = {}
            for p_base in module.predictors():
                attempts = []
                shortest_len = self.breadth
                shortest_len = min(len(evaluated_candidates[id(p_base)]),shortest_len)
                best_predictors = list(evaluated_candidates[id(p_base)].values())
                best_predictors.sort(key=lambda x: x['score'], reverse=True)

                if self.track_stats:
                    scores = [x['score'] for x in best_predictors][:10]
                    results_best[id(p_base)]["depth"].append(d)
                    results_best[id(p_base)]["max"].append(max(scores))
                    results_best[id(p_base)]["average"].append(sum(scores)/len(scores))
                    results_best[id(p_base)]["min"].append(min(scores))
                    results_best[id(p_base)]["std"].append(np.std(scores))
                
                for i in range(shortest_len-1,-1,-1):
                    attempts.append(f'Instruction #{shortest_len-i}: {best_predictors[i]["instruction"]}')
                    attempts.append(f'Prefix #{shortest_len-i}: {best_predictors[i]["prefix"]}')
                    attempts.append(f'Resulting Score #{shortest_len-i}: {best_predictors[i]["score"]}')
            
                current_metric_observations_data_boosted = [metric_observation.__str__() for metric_observation in self.metric_observation_history.current_generation_metric_observations]
                historical_metric_observation_summaries = [metric_observation.summary for metric_observation in self.metric_observation_history.previous_generations_metric_observations_used]
                if self.prompt_model: 
                    with dspy.settings.context(lm=self.prompt_model):
                        instr = dspy.Predict(GenerateInstructionGivenAttempts, n=self.breadth, temperature=self.init_temperature,model="gpt-3.5-turbo-16k")(attempted_instructions=attempts,current_metric_observations_data_boosted=current_metric_observations_data_boosted,historical_metric_observation_summaries=historical_metric_observation_summaries)
                else:
                    instr = dspy.Predict(GenerateInstructionGivenAttempts, n=self.breadth, temperature=self.init_temperature,model="gpt-3.5-turbo-16k")(attempted_instructions=attempts,current_metric_observations_data_boosted=current_metric_observations_data_boosted,historical_metric_observation_summaries=historical_metric_observation_summaries)

                if self.verbose and self.prompt_model: print(f"{self.prompt_model.inspect_history(n=1)}")
                new_candidates[id(p_base)] = instr.completions
                all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(instr.completions.proposed_prefix_for_output_field)

            if self.verbose and self.prompt_model: print(f"{self.prompt_model.inspect_history(n=1)}")
            latest_candidates = new_candidates
        
        candidates = []
        for predictor in module.predictors():
            candidates.extend(list(evaluated_candidates[id(predictor)].values()))

            if self.track_stats:
                best_predictors = list(evaluated_candidates[id(predictor)].values())
                best_predictors.sort(key=lambda x: x['score'], reverse=True)

                scores = [x['score'] for x in best_predictors][:10]
                results_best[id(predictor)]["depth"].append(d)
                results_best[id(predictor)]["max"].append(max(scores))
                results_best[id(predictor)]["average"].append(sum(scores)/len(scores))
                results_best[id(predictor)]["min"].append(min(scores))
                results_best[id(predictor)]["std"].append(np.std(scores))

        candidates.sort(key=lambda x: x['score'], reverse=True)

        candidates = self._drop_duplicates(candidates)

        best_program = candidates[0]["program"]
        best_program.candidate_programs = candidates
        best_program.total_calls = total_calls
        if self.track_stats:
            best_program.results_best = results_best
            best_program.results_latest = results_latest

        return best_program