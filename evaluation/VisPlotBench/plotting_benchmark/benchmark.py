import gc
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from plotting_benchmark.generation_engines.get_model import get_model_by_name
from .language_utils import get_language_handler

from .code_plot_generator import CodePlotGenerator
from .task_changer import TaskChanger
from .vis_generator import VisGenerator, add_index_to_filename
from .vis_judge import VisJudge
from .debug_utils import DebugSession, collect_failed_cells, update_error_rate_statistics

load_dotenv()


def get_config_template(config_folder: str | Path) -> None:
    """Copy the config template to the specified folder."""
    config_folder = Path(config_folder)
    os.makedirs(config_folder, exist_ok=True)
    resource_folder = Path(__file__).parent.resolve() / "resources"
    config_file = resource_folder / "config_template.yaml"
    shutil.copyfile(config_file, config_folder / "config_template.yaml")


def get_instructs(instruct_folder: str | Path) -> None:
    """Copy the instruction templates to the specified folder."""
    os.makedirs(instruct_folder, exist_ok=True)
    instruct_folder = Path(instruct_folder)
    resource_folder = Path(__file__).parent.resolve() / "resources"
    config_file = resource_folder / "instructs.json"
    shutil.copyfile(config_file, instruct_folder / "instructs.json")


class PlottingBenchmark:
    """
    Main benchmark class for evaluating plotting code generation capabilities.
    
    Supports both normal evaluation and self-debug modes.
    """
    
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: DictConfig | None = None,
        task_changer: TaskChanger | None = None,
    ):
        """
        Initialize the plotting benchmark.
        
        Args:
            config_path: Path to configuration file
            config: Configuration object (alternative to config_path)
            task_changer: Custom task changer for modifying prompts
        """
        self.resource_folder = Path(__file__).parent.resolve() / "resources"
        if task_changer is None:
            task_changer = TaskChanger()
        if config_path is not None:
            config = OmegaConf.load(config_path)
        elif config is None:
            raise ValueError("Provide either config or config path")
        self.config = config
        paths = self.config.paths
        self.error_rate_file = Path(paths.error_rate_file)
        benchmark_types = config.benchmark_types
        self.model_names = self.config.model_plot_gen.names

        out_folder = Path(paths.out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        self.output_file = self.get_unique_filename(out_folder, "current_results.jsonl")
        self.bench_stat_file = out_folder / paths.bench_stat_filename

        self.plotting_language = config.plotting_language
        if ("instructs_file" not in paths) or (paths.instructs_file is None):
            paths.instructs_file = self.resource_folder / "instructs.json"

        with open(paths.instructs_file, "r") as f:
            all_instructs = json.load(f)
        
        if self.plotting_language in all_instructs:
            self.instructs = all_instructs[self.plotting_language]
        else:
            raise ValueError(f"No instructions found for language '{self.plotting_language}' in instructs file")
        
        self.system_prompt = self.instructs["system_prompt"]
        self.dataset = load_dataset("TIGER-Lab/VisPlotBench", name=self.plotting_language, split="test")
        self.dataset = self._convert_image_format(self.dataset)

        self.model_judge = get_model_by_name(
            self.config.model_judge.name,
            dict(self.config.model_judge.parameters),
            self.system_prompt,
        )

        self.language_handler = get_language_handler(
            language=self.plotting_language,
            config=dict(self.config)
        )
        self.task_changer = task_changer
        self.task_changer.init_task_changer(
            data_instruct=self.instructs["data_instruct"],
            setup_instruct=self.instructs["setup_instruct"],
            data_descriptor_name=self.config.data_descriptor,
            plotting_language=self.plotting_language
        )
        dataset_folder = self.unpack_csv(paths.dataset_folder, self.dataset, self.plotting_language)

        self.plot_generator = VisGenerator(
            output_folder=out_folder,
            dataset=self.dataset,
            csv_folder=dataset_folder,
            config=self.config,
            language_handler=self.language_handler,
        )

        self.judge = VisJudge(
            vis_judge_model=self.model_judge,
            instructs=self.instructs,
            benchmark_types=benchmark_types,
            plotting_language=self.plotting_language
        )

        self.responses = None
        self.plot_responses = None

    def _convert_image_format(self, dataset):
        """Convert HuggingFace image format to the expected framework format."""
        import base64
        from io import BytesIO
        
        def convert_image_to_base64(example):
            if 'image' in example and example['image'] is not None:
                # HuggingFace images are PIL Image objects
                img = example['image']
                
                # Convert to base64 string
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Create the plots_gt field (list format, as the framework expects multiple images)
                example['plots_gt'] = [img_base64]
                
                # Remove the original image field to avoid serialization issues
                del example['image']
            else:
                example['plots_gt'] = []
            
            return example
        
        return dataset.map(convert_image_to_base64)
    
    def init_gen_model(self, model_name: str):
        """Initialize the plot generation model."""
        print(f"Plotting model parameters: {self.config.model_plot_gen.parameters}")
        self.model_plot = get_model_by_name(
            model_name,
            dict(self.config.model_plot_gen.parameters),
            self.system_prompt,
        )

        self.code_generator = CodePlotGenerator(
            model=self.model_plot,
            output_file=self.output_file,
            plotting_prompt=self.instructs["plot_instruct"],
            system_prompt=self.system_prompt,
            language_handler=self.language_handler,
        )

    @staticmethod
    def unpack_csv(csv_folder: str, dataset: Dataset, plotting_language: str) -> Path:
        """Extract CSV files from dataset to local folder."""
        if plotting_language == "mermaid" or plotting_language == "lilypond" \
        or plotting_language == "svg" or plotting_language == "asymptote":
            return None
        
        csv_folder = Path(csv_folder) / plotting_language
        csv_folder.mkdir(parents=True, exist_ok=True)

        for item in dataset:
            data_csv_content = item["data"]
            csv_path = csv_folder / f"{item['id']}.csv"

            if not csv_path.exists():
                with open(csv_path, "w") as file:
                    file.write(data_csv_content)
        print(f"CSV files unpacked to {csv_folder}")

        return csv_folder.resolve()

    @staticmethod
    def get_unique_filename(folder: Path, filename: str) -> Path:
        """Generate a unique filename if the original already exists."""
        base, extension = os.path.splitext(filename)
        i = 1
        while (folder / filename).exists():
            filename = f"{base}_{i}{extension}"
            i += 1

        return folder / filename

    def kill_vllm(self):
        """Clean up vLLM model resources."""
        del self.model_plot.llm
        del self.model_plot
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        print("Killed vLLM instance")

    def dump_results(self, dataset: pd.DataFrame) -> None:
        """Save results to JSON file."""
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        dataset.to_json(self.results_file)
        print(f"Results dumped to {self.results_file}")

    def load_results(self, ids: list[int] | None = None) -> Dataset:
        """Load results from JSON file with optional ID filtering."""
        dataset_df = pd.read_json(self.results_file)
        if isinstance(ids, list):
            dataset_df = dataset_df.loc[dataset_df["id"].isin(ids)]
        elif isinstance(ids, int):
            dataset_df = dataset_df.sample(n=ids, random_state=42)

        return dataset_df

    def run_self_debug(self, dataset_df: pd.DataFrame, model_name: str, plotting_language: str, 
                       only_stats: bool = False, skip_score: bool = False):
        """
        Execute self-debug mode for failed plotting cases.
        
        Args:
            dataset_df: Dataset with initial evaluation results
            model_name: Name of the model to use for debugging
            only_stats: If True, only calculate statistics without scoring
            skip_score: If True, skip scoring and only return dataset
        
        Returns:
            Tuple of (dataset_df, benchmark_stats)
        """
        print(f"[DEBUG] Running self debug mode for {model_name}")
        
        debug_session = DebugSession(
            model_name=model_name,
            plotting_language=plotting_language,
            output_dir=self.config.debug.output_dir,
            max_attempts=self.config.debug.top_k
        )

        initial_failed_df = collect_failed_cells(dataset_df)
        print(f"[DEBUG] Initially found {len(initial_failed_df)} failed cases that need debugging")

        # Initialize debug info structure
        if "debug_info" in dataset_df.columns:
            debug_info_map = dataset_df.set_index("id")["debug_info"].to_dict()
        else:
            debug_info_map = {
                row["id"]: {"attempts": {}} 
                for _, row in initial_failed_df.iterrows()
            }

        remaining_failed_df = initial_failed_df.copy()

        # Main debug loop
        for attempt_id in range(debug_session.max_attempts):
            print(f"[DEBUG] Starting attempt {attempt_id + 1}/{debug_session.max_attempts}")
            
            # Check for cases fixed in previous attempt
            if attempt_id > 0:
                fixed_ids = []
                for _, row in remaining_failed_df.iterrows():
                    item_id = row["id"]
                    debug_info = debug_info_map.get(item_id)
                    if debug_info is not None:
                        prev_attempt = debug_info["attempts"].get(str(attempt_id - 1))
                        if prev_attempt and prev_attempt["error"] == "" and prev_attempt["has_plot"]:
                            fixed_ids.append(item_id)
            
                if fixed_ids:
                    remaining_failed_df = remaining_failed_df[~remaining_failed_df["id"].isin(fixed_ids)]
                    print(f"[DEBUG] {len(fixed_ids)} cases were fixed in previous attempt")
            
            # Collect cases that need processing in current attempt
            current_failed_items = []
            for _, row in remaining_failed_df.iterrows():
                item_id = row["id"]
                # Ensure debug_info exists
                if debug_info_map[item_id] is None:
                    debug_info_map[item_id] = {"attempts": {}}
                
                debug_info = debug_info_map[item_id]
                
                if str(attempt_id) not in debug_info["attempts"]:
                    current_failed_items.append(row)
                
            current_failed_df = pd.DataFrame(current_failed_items) if current_failed_items else pd.DataFrame()

            if len(current_failed_df) == 0:
                print(f"[DEBUG] No remaining cases to fix in attempt {attempt_id + 1}")
                continue
            
            print(f"[DEBUG] Found {len(current_failed_df)} cases to fix in attempt {attempt_id + 1}")
            
            # Initialize model if needed
            if not hasattr(self, 'model_plot'):
                print("[INFO] Model not initialized. Initializing now...")
                self.init_gen_model(model_name)
            
            # Generate debug conversations
            debug_conversations = []
            for _, row in current_failed_df.iterrows():
                item_id = row["id"]
                attempts = debug_info_map[item_id]["attempts"]
                
                previous_attempts = [
                    attempts[str(i)] 
                    for i in range(attempt_id) 
                    if str(i) in attempts
                ]

                conversation = debug_session.generate_self_debug_conversation(row, attempt_id, previous_attempts)
                
                # Determine original error information
                if attempt_id == 0:
                    original_error = row["error"]
                    original_has_plot = row["has_plot"]
                else:
                    prev_attempt = attempts[str(attempt_id - 1)]
                    original_error = prev_attempt["error"]
                    original_has_plot = prev_attempt["has_plot"]
                
                attempts[str(attempt_id)] = {
                    "original_error": original_error,
                    "original_has_plot": original_has_plot,
                    "debug_conversation": conversation,
                    "model_response": None,
                    "code": "",
                    "error": "",
                    "has_plot": False,
                    "plots_generated": []
                }

                debug_conversations.append((item_id, conversation))

            if not debug_conversations:
                print(f"[DEBUG] No cases need processing for attempt {attempt_id}")
                continue

            # Call model for debug
            all_messages = [conv for _, conv in debug_conversations]
            responses = self.model_plot.make_self_debug_request(all_messages)

            # Process model responses
            for i, (item_id, _) in enumerate(debug_conversations):
                response = responses["response"][i]
                attempts = debug_info_map[item_id]["attempts"]
                attempts[str(attempt_id)]["model_response"] = response
                attempts[str(attempt_id)]["code"] = self.language_handler.extract_plotting_code(response)

            # Execute fixed code
            debug_rows = []
            for _, row in current_failed_df.iterrows():
                item_id = row["id"]
                code = debug_info_map[item_id]["attempts"][str(attempt_id)]["code"]
                new_row = row.copy()
                new_row["code"] = code
                debug_rows.append(new_row)

            if debug_rows:
                debug_df = pd.DataFrame(debug_rows)
                debug_df = self.plot_generator.draw_self_debug_plots(debug_df, attempt_id=attempt_id)

                # Update execution results
                for _, row in debug_df.iterrows():
                    item_id = row["id"]
                    attempt_info = debug_info_map[item_id]["attempts"][str(attempt_id)]
                    if attempt_info["model_response"] == "":
                        attempt_info.update({
                            "error": "Unexpected Error",
                            "has_plot": False,
                            "plots_generated": []
                        })
                    else:
                        attempt_info.update({
                            "error": row["error"],
                            "has_plot": row["has_plot"],
                            "plots_generated": row.get("plots_generated", [])
                        })
                    if row.get("has_plot") == False and row.get("error") == "":
                        attempt_info.update({
                            "error": "Unexpected Error",
                            "has_plot": False,
                            "plots_generated": []
                        })
            
            # Save intermediate results
            dataset_df["debug_info"] = dataset_df["id"].apply(lambda x: debug_info_map.get(x, None))
            self.dump_results(dataset_df)
            
        # Clean up resources
        if hasattr(self, 'model_plot') and self.model_plot.__class__.__name__ == "VllmEngine":
            self.kill_vllm()

        # Update error statistics
        update_error_rate_statistics(
            error_rate_file=self.error_rate_file,
            model_name=model_name,
            plotting_language=self.config.plotting_language,
            dataset_df=dataset_df
        )
        
        # Handle scoring based on parameters
        if only_stats:
            bench_stats = self.judge.calculate_self_debug_stats(dataset_df)
            with open(self.bench_stat_file, "a") as f:
                json.dump(bench_stats, f)
                f.write("\n")
            print(f"Benchmark stats saved in {self.bench_stat_file}")
            return dataset_df, bench_stats
        
        if not skip_score:
            print("[DEBUG] Calculating scores for debug attempts...")
            dataset_df = self.judge.score_self_debug(dataset_df)
            self.dump_results(dataset_df)
            bench_stats = self.judge.calculate_self_debug_stats(dataset_df)
            with open(self.bench_stat_file, "a") as f:
                json.dump(bench_stats, f)
                f.write("\n")
            print(f"Benchmark stats saved in {self.bench_stat_file}")
            return dataset_df, bench_stats
        
        return dataset_df, {}

    def run_benchmark_model(
        self,
        model_name: str,
        ids: list[int] | int | None = None,
        reuse_results: bool = False,
        load_intermediate: bool = False,
        only_stats: bool = False,
        skip_draw: bool = False,
        skip_score: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run benchmark for a single model.
        
        Args:
            model_name: Name of the model to benchmark
            ids: Specific datapoint IDs to evaluate (None for all)
            reuse_results: Whether to reuse existing results
            load_intermediate: Whether to load intermediate results
            only_stats: If True, only calculate statistics
            skip_score: If True, skip scoring step
            
        Returns:
            Tuple of (results_dataframe, benchmark_statistics)
        """

        if self.config.get("run_mode", "normal") == "self_debug":
            plotting_language = (self.config.plotting_language).split(" ")[0]
            gen_model_name = "_" + model_name.split("/")[-1]
            results_file_spostfix = (
                gen_model_name + "_" + plotting_language + "_" + self.config.data_descriptor
            )
            _, old_results_file = add_index_to_filename(
                self.config.paths.out_folder,
                self.config.paths.results_filename,
                results_file_spostfix,
            )
            
            if old_results_file is not None and os.path.exists(old_results_file):
                self.results_file = old_results_file
                print(f"[DEBUG] Loading results from {self.results_file}")
                dataset_df = self.load_results(ids)
                return self.run_self_debug(dataset_df, model_name=model_name, 
                                           plotting_language=plotting_language,
                                           only_stats=only_stats, 
                                           skip_score=skip_score)
            else:
                # First run normal mode, then debug
                self.config.run_mode = "normal"
                self.run_benchmark_model(model_name, ids, reuse_results=reuse_results, 
                                                    load_intermediate=load_intermediate, 
                                                    only_stats=only_stats, 
                                                    skip_score=skip_score)
                self.config.run_mode = "self_debug"
                dataset_df = self.load_results(ids)
                return self.run_self_debug(dataset_df, model_name=model_name, 
                                           plotting_language=plotting_language,
                                           only_stats=only_stats, 
                                           skip_score=skip_score)

        print(20 * "-")
        print(f"Benchmarking {model_name} model")
        print(20 * "-")
        gen_model_name = "_" + model_name.split("/")[-1]
        plotting_language = (self.config.plotting_language).split(" ")[0]
        results_file_spostfix = (
            gen_model_name + "_" + plotting_language + "_" + self.config.data_descriptor
        )
        new_results_file, old_results_file = add_index_to_filename(
            self.config.paths.out_folder,
            self.config.paths.results_filename,
            results_file_spostfix,
        )

        if reuse_results:
            self.results_file = old_results_file
            print(f"Loading {self.results_file}")
            dataset_df = self.load_results(ids)
        else:
            self.init_gen_model(model_name)
            self.results_file = new_results_file
            if isinstance(ids, list):
                self.dataset = self.dataset.select(ids)
            elif isinstance(ids, int):
                self.dataset = self.dataset.shuffle(seed=42).select(range(ids))
            dataset_df = self.dataset.to_pandas()
            dataset_df = self.task_changer.change_task(dataset_df)
            self.dataset = Dataset.from_pandas(dataset_df)
            dataset_df = self.code_generator.generate_codeplot_datapoints(
                self.dataset, load_intermediate
            )
            self.dump_results(dataset_df)
            
            # Clean up vLLM model resources
            if self.model_plot.__class__.__name__ == "VllmEngine":
                self.kill_vllm()
        
        if only_stats:
            bench_stats = self.judge.calculate_stats(dataset_df)
            with open(self.bench_stat_file, "a") as f:
                json.dump(bench_stats, f)
                f.write("\n")
            print(f"Benchmark stats saved in {self.bench_stat_file}")
            return dataset_df, bench_stats
        if not skip_draw:
            print("[INFO] Drawing plots...")
            dataset_df = self.plot_generator.draw_plots(dataset_df)
            
            # Handle cases with missing responses
            for index, row in dataset_df.iterrows():
                if row.get("model_response") == "":
                    dataset_df.at[index, "error"] = "Unexpected Error"
                if row.get("has_plot") == False and row.get("error") == "":
                    dataset_df.at[index, "error"] = "Unexpected Error"
                    
            self.dump_results(dataset_df)
            
            # Calculate error statistics
            total_items = len(dataset_df)
            execution_error_num = (dataset_df["error"] != "").sum()
            incorrect_plot_num = (dataset_df["has_plot"] == False).sum()

            error_rate_record_file = self.error_rate_file
            if error_rate_record_file.exists():
                with open(error_rate_record_file, "r") as f:
                    error_rates = json.load(f)
            else:
                error_rates = {}

            record_key = f"{model_name}_{plotting_language}"
            error_rates[record_key] = {
                "total_num": int(total_items),
                "execution_error_num": int(execution_error_num),
                "incorrect_plot_num": int(incorrect_plot_num)
            }

            with open(error_rate_record_file, "w") as f:
                json.dump(error_rates, f, indent=4)

        if not skip_score:
            print("[DEBUG] Calculating scores...")
            dataset_df = self.judge.score(dataset_df)
            self.dump_results(dataset_df)
            bench_stats = self.judge.calculate_stats(dataset_df)
            with open(self.bench_stat_file, "a") as f:
                json.dump(bench_stats, f)
                f.write("\n")
            print(f"Benchmark stats saved in {self.bench_stat_file}")
            return dataset_df, bench_stats
        else:
            return dataset_df, {}

    def run_benchmark(
        self,
        ids: list[int] | int | None = None,
        reuse_results: bool = False,
        load_intermediate: bool = False,
        only_stats: bool = False,
        skip_draw: bool = False,
        skip_score: bool = False,
    ) -> None:
        """
        Run benchmark for all configured models.
        
        Args:
            ids: Specific datapoint IDs to evaluate
            reuse_results: Whether to reuse existing results
            load_intermediate: Whether to load intermediate results
            only_stats: If True, only calculate statistics
            skip_score: If True, skip scoring step
        """
        for model_name in self.model_names:
            self.run_benchmark_model(
                model_name,
                ids,
                reuse_results,
                load_intermediate,
                only_stats,
                skip_draw,
                skip_score,
            )
