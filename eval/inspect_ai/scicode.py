import copy
import time
import shutil
import subprocess
import tempfile
import asyncio
import atexit
import json
import sys
import os
from typing import Any
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.scorer import scorer, mean, metric, Metric, Score, Target
from scicode.parse.parse import extract_function_name, get_function_from_code
from scicode.gen.models import generate_dummy_response, extract_python_script

BACKGOUND_PROMPT_TEMPLATE = Path("../data", "multistep_template.txt").read_text()
DEFAULT_PROMPT_TEMPLATE = Path("../data", "background_comment_template.txt").read_text()

class ScicodePromptingAssistant:
    def __init__(
        self,
        output_dir: Path,
        prompt_dir: Path,
        with_background: bool,
    ):
        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.with_background = with_background
        self.previous_llm_code = []
        
    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"
    
    def register_previous_response(
        self,
        prob_data: dict,
        response: str,
        previous_code: str,
        num_steps: int,
    ):
        self.previous_llm_code[num_steps - 1] = extract_python_script(response)
        self.save_response_with_steps(
            prob_data,
            response,
            previous_code,
            num_steps,
        )
    
    def save_response_with_steps(
        self, 
        prob_data: dict, 
        response: str, 
        previous_code: str, 
        num_steps: int
    ) -> None:
        output_dir = Path(
            self.output_dir,
            self._get_background_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        output_file_path = output_dir / f"{prob_id}.{num_steps}.py"
        python_code = extract_python_script(response)
        output_file_path.write_text(f'{previous_code}\n{python_code}', encoding="utf-8")    
    
    @staticmethod
    def process_problem_code(
        prob_data: dict, 
        num_steps: int
    ) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string
    
    def process_problem_steps(
        self, 
        problem_data: dict, 
        num_steps: int
    ):
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        previous_code = []
        for i in range(num_steps - 1):
            output_lines.append(problem_data["sub_steps"][i]["step_description_prompt"] + '\n' +
                                problem_data["sub_steps"][i]["step_background"] if self.with_background
                                else problem_data["sub_steps"][i]["step_description_prompt"])
            code_content = self.previous_llm_code[i] if self.previous_llm_code[i] is not None else "# Code missing"
            output_lines.append(code_content)
            previous_code.append(code_content)
            output_lines.append("------")

        next_step.append(problem_data["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                         problem_data["sub_steps"][num_steps - 1]["step_background"] if self.with_background
                         else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"])
        next_step.append(self.process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str
    
    def generate_prompt_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    ):
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = self.process_problem_steps(prob_data, num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n'
    
    def save_prompt_with_steps(
            self, 
            prob_data: dict, 
            prompt: str, 
            num_steps: int
        ) -> None:
        output_dir = Path(
            self.prompt_dir, 
            self._get_background_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def prepare_final_prompt_with_steps(
        self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        *,
        save: bool = True
    ):
        prob_id = prob_data["problem_id"]
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (
                        (prob_id == "13" and prev_step == 5) or 
                        (prob_id == "62" and prev_step == 0) or 
                        (prob_id == "76" and prev_step == 2)
                    ):
                        prev_file_path = Path(
                            "../data",
                            f"{prob_id}.{prev_step+1}.txt"
                        )
                    else:
                        prev_file_path = Path(
                            self.output_dir,
                            self._get_background_dir(),
                            f"{prob_id}.{prev_step + 1}.py"
                        )
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding='utf-8')
                        func_name = extract_function_name(
                            prob_data["sub_steps"][prev_step]["function_header"]
                        )
                        function_code = get_function_from_code(
                            prev_file_content, func_name
                        )
                        self.previous_llm_code[prev_step] = function_code if function_code else "# Function not found"
                    else:
                        raise Exception(f'Generating problem {prob_id} step {num_steps} ahead of step {prev_step + 1}.')
                
        prompt, previous_code = self.generate_prompt_with_steps(
            prob_data,
            num_steps,
            prompt_template,
        )
        if save:
            self.save_prompt_with_steps(
                prob_data,
                prompt,
                num_steps,
            )
        return prompt, previous_code

class ScicodeEvaluator:
    def __init__(
        self,
        h5py_file: str,
        code_dir: Path,
        log_dir: Path,
        with_background: bool,
    ):
        self.h5py_file = h5py_file
        self.code_dir = code_dir
        self.log_dir = log_dir
        self.with_background = with_background
    
    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"
        
    def test_code(
        self,
        prob_data: dict,
    ):
        code_dir = Path(
            self.code_dir,
            "generated_code",
            self._get_background_dir()
        )
        tmp_dir = Path(tempfile.mkdtemp())
        
        sub_steps = prob_data["sub_steps"]
        problem_id = prob_data["problem_id"]
        for idx in range(len(sub_steps)):
            if (
                (problem_id == "13" and idx == 5) or
                (problem_id == "62" and idx == 0) or
                (problem_id == "76" and idx == 2)
            ):
                continue
            step_id = sub_steps[idx]["step_number"]
            code_file_path = Path(code_dir, f"{step_id}.py")
            assert code_file_path.is_file(), f"Code file {code_file_path} not found."
            code_content = code_file_path.read_text(encoding='utf-8')
            test_lst = sub_steps[idx]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(f"""

from scicode.parse.parse import process_hdf5_to_tuple

""")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)}, '{self.h5py_file}')" + '\n')
                for i in range(len(test_lst)):
                    f.write(f"target = targets[{i}]\n\n")
                    for line in test_lst[i].split('\n'):
                        f.write(line + '\n')
                        
        def run_script(script_path):
            try:
                subprocess.run(['python', script_path], check=True, capture_output=True,
                            text=True, timeout=1800)
                return 0
            except subprocess.CalledProcessError:
                return 1
            except subprocess.TimeoutExpired:
                return 2
            
        total_steps = len(sub_steps)
        total_correct = 0
        for idx in range(len(sub_steps)):
            if (
                (problem_id == "13" and idx == 5) or
                (problem_id == "62" and idx == 0) or
                (problem_id == "76" and idx == 2)
            ):
                continue
            step_id = sub_steps[idx]["step_number"]
            script_path = Path(tmp_dir, f'{step_id}.py')
            logs_dir = Path(
                self.log_dir,
                "evaluation_logs",
                self._get_background_dir()
            )
            logs_dir.mkdir(parents=True, exist_ok=True)
            logs_file = Path(
                logs_dir,
                f"{step_id}.log"
            )
            if logs_file.is_file():
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        total_correct += 1
                continue
            ret = run_script(script_path)
            if ret == 0:
                with open(logs_file, 'w') as f:
                    f.write('pass')
                total_correct += 1
            elif ret == 1:
                with open(logs_file, 'w') as f:
                    f.write('fail')
            else:
                with open(logs_file, 'w') as f:
                    f.write('time out')
        
        shutil.rmtree(tmp_dir)
        problem_correct = 1 if total_correct == total_steps else 0
        return problem_correct, total_correct, total_steps

def record_to_sample(record):
    return Sample(
        input="problem_id",
        target=record["problem_id"],
        id=record["problem_id"],
        metadata={
            k: v for k, v in record.items()
        }
    )

def generate_gold_response(prob_data: dict, num_steps: int):
    return f"Blah blah\n```python\n{prob_data['sub_steps'][num_steps - 1]['ground_truth_code']}\n```\n"

@solver
def scicode_solver(**params: dict[str, Any]):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_name = str(state.model).replace("/", "-")
        # Ensure timestamp is consistent across the run by using state.epoch or passed param, 
        # but here we can just use the current run's start time if available or generate one.
        # However, to be consistent for all samples in this run, we need a stable timestamp.
        # inspect_ai doesn't pass a run-level timestamp easily to solver, but we can assume
        # the user wants a unique folder per run.
        # A simple way is to use the run_id from state.
        
        # Define base output directory: results/model_name/timestamp_runid/
        # We'll use a formatted timestamp + run_id prefix to ensure uniqueness and sorting
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # To avoid different timestamps for different samples, we should probably set this once.
        # But `solve` is called per sample. 
        # Better approach: Use the `output_dir` passed in params if it already contains the structure,
        # OR construct it here. The user requested "results/model_name/timestamp".
        # Since we can't easily share a timestamp across distributed solvers without a coordinator,
        # we might use the run_id which is unique per run.
        # But the user specifically asked for timestamp. 
        # Let's check if we can reuse the global log timestamp logic or if we should just use run_id.
        # Given the constraints, let's use a module-level variable to store the run timestamp
        # initialized when the module is loaded or first called.
        
        global RUN_TIMESTAMP
        if 'RUN_TIMESTAMP' not in globals():
             RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        
        # Use state.sample_id or a fallback for uniqueness if run_id is missing, 
        # but better yet, just use the timestamp since run_id isn't available on TaskState
        # in this version.
        run_output_dir = Path("results", model_name, f"{RUN_TIMESTAMP}")
        
        prompt_assistant = ScicodePromptingAssistant(
            output_dir=Path(run_output_dir, "generated_code"),
            prompt_dir=Path(run_output_dir, "prompt"),
            with_background=params["with_background"],
        )
        prompt_template = BACKGOUND_PROMPT_TEMPLATE if params["with_background"] else DEFAULT_PROMPT_TEMPLATE
        sub_steps = state.metadata["sub_steps"]
        for idx in range(len(sub_steps)):
            prob_id = state.metadata["problem_id"]
            if (
                (prob_id == "13" and idx == 5) or
                (prob_id == "62" and idx == 0) or
                (prob_id == "76" and idx == 2)
            ):
                continue
            prompt, previous_code = prompt_assistant.prepare_final_prompt_with_steps(
                prob_data=state.metadata,
                num_steps=idx+1,
                tot_steps=len(sub_steps),
                prompt_template=prompt_template,
            )
            
            if params["mode"] == "dummy":
                response_from_llm = generate_dummy_response(prompt)
            elif params["mode"] == "gold":
                response_from_llm = generate_gold_response(state.metadata, idx+1)
            else:
                try:
                    # ===Model Generation===
                    state.user_prompt.text = prompt
                    state_copy = copy.deepcopy(state)
                    result = await generate(state=state_copy)
                    response_from_llm = result.output.completion
                    # ===Model Generation===
                except:
                    print(f"Failed to generate response for problem {prob_id} step {idx+1}.")
                    response_from_llm = generate_dummy_response(prompt)
            prompt_assistant.register_previous_response(
                prob_data=state.metadata,
                response=response_from_llm,
                previous_code=previous_code,
                num_steps=idx+1,
            )
        return state
    return solve

@metric
def sub_problem_correctness() -> Metric:
    def metric(scores: list[Score]) -> int | float:
        total_correct = 0
        total_steps = 0
        for score in scores:
            total_correct += score.value["Total Correct"]
            total_steps += score.value["Total Steps"]
        return total_correct / total_steps
    return metric

@scorer(
    metrics=[{
        "Problem Correctness": [mean()],
    }, sub_problem_correctness()]
)
def scicode_scorer(**params: dict[str, Any]):
    async def score(state: TaskState, target: Target):
        model_name = str(state.model).replace("/", "-")
        # Reconstruct the same directory structure as in solver
        # Note: RUN_TIMESTAMP must be consistent. In a single process run it works.
        # For safety in distributed/restarted envs, we should rely on run_id or passed params.
        # But here we use the same global approach for consistency within the same process.
        global RUN_TIMESTAMP
        if 'RUN_TIMESTAMP' not in globals():
             import datetime
             RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        
        run_output_dir = Path("results", model_name, f"{RUN_TIMESTAMP}")
        
        evaluator = ScicodeEvaluator(
            h5py_file=params["h5py_file"],
            code_dir=run_output_dir,
            log_dir=run_output_dir,
            with_background=params["with_background"],
        )
        loop = asyncio.get_running_loop()
        problem_correct, total_correct, total_steps = await loop.run_in_executor(
            None, evaluator.test_code, state.metadata
        )
        return Score(
            value={
                "Problem Correctness": problem_correct,
                "Total Correct": total_correct,
                "Total Steps": total_steps,
            }
        )
    return score

@task
def scicode(
    split: str = 'test',
    output_dir: str = './tmp',
    with_background: bool = False,
    h5py_file: str = '../data/test_data.h5',
    mode: str = 'normal',
):
    
    dataset =  hf_dataset(
        'SciCode1/SciCode',
        split=split,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=scicode_solver(
            output_dir=output_dir,
            with_background=with_background,
            mode=mode,
        ),
        scorer=scicode_scorer(
            output_dir=output_dir,
            with_background=with_background,
            h5py_file=h5py_file,
        ),
    )

def auto_rename_inspect_logs():
    """Automatically rename inspect logs to scicode_{model}_{timestamp}.json format on exit."""
    try:
        # 1. Determine log directory priority: --log-dir arg > INSPECT_LOG_DIR env > ./logs
        log_dir = Path("logs")
        
        # Check environment variable
        if "INSPECT_LOG_DIR" in os.environ:
            log_dir = Path(os.environ["INSPECT_LOG_DIR"])
            
        # Check command line arguments
        args = sys.argv
        for i in range(len(args) - 1):
            if args[i] == "--log-dir":
                log_dir = Path(args[i+1])
                break
        
        if not log_dir.exists():
            return

        # 2. Find the latest JSON log file related to scicode
        # Look for files created in the last 20 seconds to avoid renaming old logs
        json_files = list(log_dir.glob("*_scicode_*.json"))
        if not json_files:
            return
            
        # Sort by modification time, newest first
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_log = json_files[0]
        
        # Ensure the log is recent (e.g. within 20 seconds)
        if time.time() - latest_log.stat().st_mtime > 20:
            return

        # --- NEW: Check for eval-retry mode ---
        is_retry = False
        retry_source_log = None
        
        # Check if 'eval-retry' is in arguments
        if "eval-retry" in sys.argv:
            is_retry = True
            # Find the source log file in arguments
            # It should be a file ending in .json or .eval that exists
            for arg in sys.argv:
                # Skip the command itself
                if arg == "eval-retry":
                    continue
                try:
                    p = Path(arg)
                    if p.is_file() and p.suffix in ['.json', '.eval']:
                        retry_source_log = p.resolve()
                        break
                except Exception:
                    continue
        
        if is_retry and retry_source_log:
            # In retry mode, overwrite the original log with the new one
            try:
                if latest_log.resolve() != retry_source_log:
                    # Replace original with new
                    latest_log.replace(retry_source_log)
                    print(f"\n[Auto-Rename] Retry detected. Updated original log: {retry_source_log}")
                    return
            except Exception as e:
                print(f"\n[Auto-Rename] Failed to update original log: {e}")
                # Fall through to normal rename if update fails
        # --------------------------------------

        # 3. Read content to get model and timestamp
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it's a valid inspect log with expected fields
                if "eval" not in data:
                    return
                model = data['eval']['model']
                timestamp = data['eval']['created']
                
            # 4. Construct new filename
            # Handle timestamp if it's not a string (though json load usually gives string)
            if not isinstance(timestamp, str):
                timestamp = str(timestamp)
            
            safe_model = str(model).replace("/", "-")
            safe_timestamp = timestamp.replace(":", "-")
            
            new_filename = f"scicode_{safe_model}_{safe_timestamp}.json"
            new_path = log_dir / new_filename
            
            # 5. Rename if name is different
            if latest_log.name != new_filename:
                latest_log.rename(new_path)
                print(f"\n[Auto-Rename] Log renamed to: {new_path}")
                
        except Exception:
            # Silently fail if reading/parsing fails to avoid disrupting the user
            pass
            
    except Exception:
        pass

# Register the cleanup function
atexit.register(auto_rename_inspect_logs)

if __name__ == "__main__":
    from inspect_ai import eval
    eval(scicode, log_format="json")
