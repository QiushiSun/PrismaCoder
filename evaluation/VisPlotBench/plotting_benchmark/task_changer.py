import json
import warnings
from io import StringIO
from typing import Callable

import numpy as np
import pandas as pd


class TaskChanger:
    """
    There are four main methods

    setup_changer(task_text: str, df: pd.DataFrame) -> str: takes setup description, returns altered one
    data_descr_changer(task_text: str, df: pd.DataFrame) -> str: takes data description, dataframe, returns altered one
    plot_descr_changer(task_text: str, df: pd.DataFrame) -> str: takes plot description, returns altered one
    style_changer(task_text: str, df: pd.DataFrame) -> str: takes plot style description, returns altered one

    there are implemented default method for data description using pycharm_dataframe_description
    Other methods are dummy.

    To do experiments you need to rewrite any methods you need, inheriting this class
    """

    def __init__(
        self,
        data_descriptor_map: dict[str, Callable] = {},
    ) -> None:
        self.data_descriptor_map = {
            "pycharm": self.pycharm_df_description,
            "datalore": self.datalore_df_description,
            "lida": self.lida_df_description,
            "head": self.head_df_description,
            "describe": self.describe_df_description,
            "empty": self.empty_df_description,
        }
        self.data_descriptor_map.update(data_descriptor_map)

        self.change_mapping = {
            "setup": self.setup_changer,
            "data_description": self.data_descr_changer,
            "plot_description": self.plot_descr_changer,
            "plot_style": self.style_changer,
        }
        self.task_names = list(self.change_mapping.keys())

    def init_task_changer(
        self,
        data_descriptor_name: str,
        data_instruct: str,
        setup_instruct: str,
        plotting_language: str = None
    ) -> None:
        self.data_instruct = data_instruct
        self.setup_instruct = setup_instruct
        self.plotting_language = plotting_language
        if data_descriptor_name in self.data_descriptor_map:
            self.data_descriptor = self.data_descriptor_map[data_descriptor_name]
        else:
            raise ValueError(f"Data descriptor {data_descriptor_name} is not defined")
        self.data_descriptor_name = data_descriptor_name

    def setup_changer(
        self, task_text: str | None, df: pd.DataFrame, dp_row: pd.Series
    ) -> str:
        setup_instruct = self.setup_instruct
        if 'used_lib' in dp_row and not isinstance(dp_row['used_lib'], (list, tuple, np.ndarray)) and pd.notna(dp_row['used_lib']):
            used_lib = str(dp_row['used_lib']).strip()
            setup_instruct = setup_instruct.replace('[USED_LIB]', used_lib)
            if used_lib == 'holoviews':
                setup_instruct = setup_instruct + "Ensure that holoviews is configured with the bokeh backend by adding `hv.extension('bokeh')` at the beginning."
            elif used_lib == 'chartify':
                setup_instruct = setup_instruct + "Use `import chartify` and initialize with `chartify.Chart()`."
        return setup_instruct
        
    def data_descr_changer(
        self, task_text: str, df: pd.DataFrame, dp_row: pd.Series
    ) -> str:
        if self.plotting_language == "asymptote":
            data_content = dp_row["data"].strip()
            if data_content:
                data_description = self.data_instruct + "\n" + data_content
            else:
                data_description = "Data description. The Asymptote code is constructed only from the provided instructions. Do not load, include, or reference external files, datasets, or resources."
            return data_description[:4000]
    
        if df.empty:
            data_description = self.data_instruct
            return data_description[:4000]
        
        data_description = self.data_descriptor(df)
        data_description = self.data_instruct + "\n" + data_description
        return data_description[:4000]

    def style_changer(self, task_text: str, df: pd.DataFrame, dp_row: pd.Series) -> str:
        return task_text

    def plot_descr_changer(
        self, task_text: str, df: pd.DataFrame, dp_row: pd.Series
    ) -> str:
        return task_text

    def change_task_dp(self, dp_row: pd.Series) -> pd.Series:
        # data_df = pd.read_csv(StringIO(dp_row["data"]))
        if self.plotting_language == "asymptote":
            data_df = pd.DataFrame() 
        else:
            if dp_row["data"].strip() == "":
                data_df = pd.DataFrame()
            else:
                data_df = pd.read_csv(StringIO(dp_row["data"]))

        for task_name in self.task_names:
            row_name = f"task__{task_name}"
            changer = self.change_mapping[task_name]
            task_text = dp_row[row_name] if row_name in dp_row else None
            dp_row[row_name] = changer(task_text=task_text, df=data_df, dp_row=dp_row)

        return dp_row

    def change_task(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.apply(self.change_task_dp, axis=1)
        dataset["data_descriptor"] = len(dataset) * [self.data_descriptor_name]

        return dataset

    @staticmethod
    def pycharm_df_description(df: pd.DataFrame) -> str:
        # task_text is dummy

        descr_lines = [f"Number of rows in DataFrame: {len(df)}"]
        descr_lines.append("DataFrame has the following columns:")
        for col in df.columns:
            types_set = set(df.loc[df[col].notna(), col].apply(type))
            types_list = [str(type_.__name__) for type_ in types_set]
            if len(types_list) == 1:
                col_types = types_list.pop()
            else:
                col_types = str(set(types_list)).replace('"', "").replace("'", "")
            descr = f"{col} of type {col_types}. Count: {df[col].count()}"
            if str(df[col].dtype).startswith(("int", "float")):
                mean = f"{df[col].mean():.6}"
                std = f"{df[col].std():.6}"
                if str(df[col].dtype).startswith("int"):
                    minimum = f"{df[col].min()}"
                    maximum = f"{df[col].max()}"
                else:
                    minimum = f"{df[col].min():.6}"
                    maximum = f"{df[col].max():.6}"
                descr = (
                    descr
                    + f", Mean: {mean}, Std. Deviation: {std}, Min: {minimum}, Max: {maximum}"
                )
            descr_lines.append(descr)

        return ("\n".join(descr_lines)).lstrip()

    @staticmethod
    def datalore_df_description(df_to_descr: pd.DataFrame) -> str:
        """Example of format for titanic dataset:
        titanic_df - DataFrame with 12 columns and 891 rows
        Column names: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, ↩
        Parch, Ticket, Fare, Cabin, Embarked
        """
        descr_string = (
            f"DataFrame with {len(df_to_descr.columns)} columns "
            f"and {len(df_to_descr)} rows\n"
        )
        descr_string += "Column names and types: " + ", ".join(
            [f"{column}: {df_to_descr[column].dtype}" for column in df_to_descr.columns]
        )
        return descr_string

    @staticmethod
    def lida_df_description(df_to_descr: pd.DataFrame) -> str:
        """
        Get properties of each column in a pandas DataFrame
        borrowed from: https://github.com/microsoft/lida/blob/9bb26c0adb56cab2d7c5d49ad96bc14e204c87ec/lida/components/summarizer.py#L34
        impossible to install original library due to
        """

        # unused legacy
        n_samples = 3

        def check_type(dtype: str, value):
            if np.isnan(value):
                return 0
            """Cast value to right type to ensure it is JSON serializable"""
            if "float" in str(dtype):
                return float(value)
            elif "int" in str(dtype):
                return int(value)
            else:
                return value

        properties_list = []
        for column in df_to_descr.columns:
            dtype = df_to_descr[column].dtype
            properties = {}
            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                properties["std"] = check_type(dtype, df_to_descr[column].std())
                properties["min"] = check_type(dtype, df_to_descr[column].min())
                properties["max"] = check_type(dtype, df_to_descr[column].max())

            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                # Check if the string column can be cast to a valid datetime
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df_to_descr[column], errors="raise")
                        properties["dtype"] = "date"
                except ValueError:
                    # Check if the string column has a limited number of values
                    if df_to_descr[column].nunique() / len(df_to_descr[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df_to_descr[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df_to_descr[column]):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            # add min max if dtype is date
            if properties["dtype"] == "date":
                try:
                    properties["min"] = df_to_descr[column].min()
                    properties["max"] = df_to_descr[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df_to_descr[column], errors="coerce")
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()
            # Add additional properties to the output dictionary
            nunique = df_to_descr[column].nunique()
            if "samples" not in properties:
                non_null_values = df_to_descr[column][
                    df_to_descr[column].notnull()
                ].unique()
                n_samples = min(n_samples, len(non_null_values))
                samples = (
                    pd.Series(non_null_values)
                    .sample(n_samples, random_state=42)
                    .tolist()
                )
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            properties_list.append({"column": column, "properties": properties})

        return json.dumps(properties_list, indent=2)

    @staticmethod
    def head_df_description(df_to_descr: pd.DataFrame) -> str:
        """Get head description"""
        descr_string = (
            f"DataFrame with {len(df_to_descr.columns)} columns "
            f"and {len(df_to_descr)} rows\n"
        )
        descr_string += "Column names and types: " + ", ".join(
            [f"{column}: {df_to_descr[column].dtype}" for column in df_to_descr.columns]
        )
        descr_string += "\nHead:\n"
        with pd.option_context(
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
            "display.width",
            10000,
        ):
            head = str(df_to_descr.head().to_json())
        descr_string += head
        return descr_string

    @staticmethod
    def describe_df_description(df_to_descr: pd.DataFrame) -> str:
        """Get pandas describe method description"""
        descr_string = (
            f"DataFrame with {len(df_to_descr.columns)} columns "
            f"and {len(df_to_descr)} rows\n"
        )
        descr_string += "Column names and types: " + ", ".join(
            [f"{column}: {df_to_descr[column].dtype}" for column in df_to_descr.columns]
        )
        with pd.option_context(
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
            "display.width",
            10000,
        ):
            descr = str(df_to_descr.describe().to_json())
        descr_string += "\nColumns stats:\n" + descr
        return descr_string

    @staticmethod
    def empty_df_description(df_to_descr: pd.DataFrame) -> str:
        return ""
