# from zenml import pipeline, step


# @step
# def step_1() -> str:
#     """Returns the `world` string."""
#     return "world"


# @step(enable_cache=False)
# def step_2(input_one: str, input_two: str) -> None:
#     """Combines the two strings at its input and prints them."""
#     combined_str = f"{input_one} {input_two}"
#     print(combined_str)


# @pipeline(enable_cache=False)
# def my_pipeline():
#     output_step_one = step_1()
#     step_2(input_one="hello", input_two=output_step_one)


# if __name__ == "__main__":
#     my_pipeline()
import mlflow

experiments = mlflow.search_runs()
print(experiments)

