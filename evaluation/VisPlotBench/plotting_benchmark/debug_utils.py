import pandas as pd
from typing import List
from pathlib import Path
import json

def collect_failed_cells(df: pd.DataFrame) -> pd.DataFrame:
    # Return rows where there is an error or no plot was generated
    return df[(df["error"] != "") | (~df["has_plot"])].copy()


class DebugSession:
    def __init__(self, model_name: str, plotting_language: str, output_dir: Path, max_attempts: int = 1):
        # Initialize a debug session for a specific model and output directory
        self.model_name = model_name
        self.plotting_language = plotting_language  
        self.output_dir = Path(output_dir)
        self.max_attempts = max_attempts
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_self_debug_conversation(
        self,
        row: pd.Series,
        attempt_id: int,
        previous_attempts: List[dict] = None,
    ) -> List[dict]:
        # Generate a debug conversation for a given row and attempt
        if attempt_id == 0:
            return self._generate_single_turn_conversation(row, self.plotting_language)
        else:
            return self._generate_multi_turn_conversation(row, previous_attempts or [], self.plotting_language)
        
    @staticmethod
    def _generate_single_turn_conversation(row: pd.Series, plotting_language: str) -> List[dict]:
        # Generate a single-turn conversation for the first attempt
        task = row.get('task', '') or '\n'.join(
            str(row[f]) for f in row.index if f.startswith("task__") and pd.notna(row[f])
        )
        return [
            {"content": task},
            {"content": f"```{plotting_language}\n{row['code']}\n```", "is_assistant": True},
            {"content": f"The above code failed with the following error:\n{row['error']}\nPlease provide a corrected version enclosed in ```{plotting_language} ... ```."}
        ]

    @staticmethod
    def _generate_multi_turn_conversation(row: pd.Series, history: List[dict], plotting_language: str) -> List[dict]:
        # Generate a multi-turn conversation based on previous attempts
        conversation = history[-1]['debug_conversation'].copy()
        
        for attempt in history:
            conversation.append({"content": f"```{plotting_language}\n{attempt['code']}\n```", "is_assistant": True})
            conversation.append({"content": f"The above code still failed with the following error:\n{attempt['error']}\nPlease provide a corrected version enclosed in ```{plotting_language} ... ```."})

        return conversation

def update_error_rate_statistics(
    error_rate_file: Path,
    model_name: str,
    plotting_language: str,
    dataset_df: pd.DataFrame,
) -> None:
    """
    Update error rate statistics and write them to a file.
    """
    if error_rate_file.exists():
        # Load existing error rate statistics if the file exists
        with open(error_rate_file, "r") as f:
            error_rates = json.load(f)
    else:
        error_rates = {}
    
    if "debug_info" not in dataset_df.columns:
        # No debug info column found in the DataFrame
        print("[DEBUG] No debug info found")
        return

    debug_cases = dataset_df[dataset_df['debug_info'].notna()]
    if len(debug_cases) == 0:
        # No debug cases found in the DataFrame
        print("[DEBUG] No debug cases found")
        return

    # Record original evaluation data for the model and plotting language
    record_key = f"{model_name}_{plotting_language.split(' ')[0]}"
    error_rates[record_key] = {
        "total_num": len(dataset_df),
        "execution_error_num": len(dataset_df[dataset_df["error"] != ""]),
        "incorrect_plot_num": len(dataset_df[~dataset_df["has_plot"]]),
    }
    
    # Get the maximum number of attempts from debug info
    max_attempts = max(
        max(int(attempt) for attempt in row['debug_info']['attempts'].keys())
        for _, row in debug_cases.iterrows()
    ) + 1
    
    # Record the result of each attempt
    debug_attempts = {}
    remaining_cases = len(debug_cases)  # Number of cases to debug at the start
    
    for attempt_idx in range(max_attempts):
        execution_error_num = 0
        incorrect_plot_num = 0
        
        # Count errors and incorrect plots for the current attempt
        for _, row in debug_cases.iterrows():
            debug_info = row['debug_info']
            attempt = str(attempt_idx)
            if attempt not in debug_info['attempts']:
                continue
                
            attempt_info = debug_info['attempts'][attempt]
            if attempt_info['error'] != "":
                execution_error_num += 1
            if not attempt_info['has_plot']:
                incorrect_plot_num += 1
        
        debug_attempts[f"attempt_{attempt_idx}"] = {
            "total_num": remaining_cases,  # Number of cases to debug in this attempt
            "execution_error_num": execution_error_num,
            "incorrect_plot_num": incorrect_plot_num
        }
        
        # Update the number of cases to debug in the next round
        remaining_cases = execution_error_num  # Only cases with execution errors need further debugging
    
    error_rates[record_key]["debug_attempts"] = debug_attempts
    
    # Save updated error rate statistics to file
    with open(error_rate_file, "w") as f:
        json.dump(error_rates, f, indent=4)
    
    print(f"[DEBUG] Error rate statistics saved to {error_rate_file}")

