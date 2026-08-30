import os
import json
from tqdm import tqdm
from test_prompt import *
from together import Together
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================================================
# Configuration
# =========================================================

json_file = "api_keys.json"

dataset_path = "historical.json"
output_path = "results"

# Relative cutoff years:
# cutoff_year - event_year
#
# Examples:
#   range(-9, 1)  -> -9, -8, ..., -1, 0
#   range(-10, 1) -> -10, -9, ..., -1, 0
#   range(-5, 1)  -> -5, -4, ..., -1, 0
RELATIVE_CUTOFF_YEARS = range(-10, 1)

# Maximum number of concurrent API requests.
# Adjust this value according to API rate limits.
MAX_WORKERS = 25


# =========================================================
# API Clients
# =========================================================

with open(
    json_file,
    "r",
    encoding="utf-8"
) as f:

    keys = json.load(f)

together_client = Together(
    api_key=keys.get("together")
)

openai_client = openai.OpenAI(
    api_key=keys.get("openai")
)


# =========================================================
# Output Directories
# =========================================================

os.makedirs(
    output_path,
    exist_ok=False
)

intermediate_dir = os.path.join(
    output_path,
    "intermediate"
)

os.makedirs(
    intermediate_dir,
    exist_ok=True
)


# =========================================================
# Models
# =========================================================

model_list = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    #"gpt-4o-2024-08-06": "openai"
}


# =========================================================
# Prompt Variants
# =========================================================

prompt_variants = {
    "P1": P1,
    "ZS_COT_PIPELINE": ZS_COT_PIPELINE,
    "PS_PIPELINE": PS_PIPELINE,
    "SR_PIPELINE": SR_PIPELINE,
    "QR_PIPELINE": QR_PIPELINE,
    "SR_QR_PIPELINE": SR_QR_PIPELINE,
    "QR_SR_PIPELINE": QR_SR_PIPELINE,
}


# =========================================================
# Utility Functions
# =========================================================

def extract_answer(text):
    """
    Extract a final Yes/No answer from the model output.

    Only outputs that exactly match "yes" or "no"
    are considered valid answers.
    """

    text = text.strip().lower()

    if text in ["yes", "no"]:
        return text.capitalize()

    return "N/A"


def run_chat_completion(
    model,
    api_type,
    system_prompt,
    user_prompt
):
    """
    Run a chat completion using the specified API provider.
    """

    if api_type == "together":

        response = together_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=1024,
            temperature=0
        )

        return response.choices[0].message.content.strip()

    elif api_type == "openai":

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=1024,
            temperature=0
        )

        return response.choices[0].message.content.strip()

    else:

        raise ValueError(
            f"Unknown API type: {api_type}"
        )


def resolve_steps(prompt_template):
    """
    Resolve a prompt template into a list of processing steps.

    Supported formats:
    - {"steps": [{system_prompt, user_prompt}, ...]}
    - [{system_prompt, user_prompt}, ...]
    - {system_prompt, user_prompt}
    """

    if (
        isinstance(prompt_template, dict)
        and "steps" in prompt_template
    ):

        steps = prompt_template["steps"]

        if not isinstance(
            steps,
            (list, tuple)
        ):

            raise ValueError(
                "prompt_template['steps'] must be "
                "a list or tuple."
            )

        return list(steps)

    if isinstance(
        prompt_template,
        (list, tuple)
    ):

        return list(prompt_template)

    if (
        isinstance(prompt_template, dict)
        and "system_prompt" in prompt_template
        and "user_prompt" in prompt_template
    ):

        return [prompt_template]

    raise ValueError(
        "Invalid prompt_template format."
    )


# =========================================================
# Process One Question
# =========================================================

def process_one(
    pair,
    model,
    api_type,
    prompt_style,
    prompt_template,
    relative_cutoff,
    inter_path
):
    """
    Process one question under one relative cutoff condition.

    The actual cutoff year is calculated as:

        cutoff_year = event_year + relative_cutoff

    For example:
        event_year = 1265
        relative_cutoff = -3
        cutoff_year = 1262

    The dataset provides two reference answers:
        - Answer before cutoff
        - Answer after cutoff
    """

    event_year = int(
        pair["Event year"]
    )

    # Calculate the actual cutoff year.
    cutoff_year = (
        event_year + relative_cutoff
    )

    # The previous cutoff year is one year earlier.
    cutoff_year_minus_1 = (
        cutoff_year - 1
    )

    steps = resolve_steps(
        prompt_template
    )

    context = {}
    step_logs = []
    output_text = None

    # Execute all steps sequentially for this question.
    for step_idx, step_template in enumerate(
        steps,
        start=1
    ):

        system_prompt = step_template[
            "system_prompt"
        ].format(
            cutoff_year=cutoff_year,
            cutoff_year_minus_1=cutoff_year_minus_1,
            **context
        )

        user_prompt = step_template[
            "user_prompt"
        ].format(
            QUESTION=pair["Question"],
            cutoff_year=cutoff_year,
            cutoff_year_minus_1=cutoff_year_minus_1,
            **context
        )

        output_text = run_chat_completion(
            model,
            api_type,
            system_prompt,
            user_prompt
        )

        # Pass the current step's output
        # to subsequent steps.
        context["REASONING"] = output_text

        context[
            f"STEP{step_idx}_OUTPUT"
        ] = output_text

        step_logs.append(
            {
                "step": step_idx,
                "output": output_text
            }
        )

    extracted_answer = extract_answer(
        output_text
    )

    # Save the intermediate result.
    with open(
        inter_path,
        "a",
        encoding="utf-8"
    ) as wf:

        wf.write(
            json.dumps(
                {
                    "event_year": event_year,
                    "cutoff_year": cutoff_year,
                    "relative_cutoff": relative_cutoff,
                    "question": pair["Question"],
                    "steps": step_logs,
                    "model_output": output_text,
                    "extracted_answer": extracted_answer
                },
                ensure_ascii=False
            )
            + "\n"
        )

    # Store the final result.
    result = {
        "question": pair["Question"],
        "answer_before_cutoff": pair.get(
            "Answer before cutoff",
            "N/A"
        ),
        "answer_after_cutoff": pair.get(
            "Answer after cutoff",
            "N/A"
        ),
        "intermediate_steps": step_logs,
        "Event year": event_year,
        "Cutoff year": cutoff_year,
        "relative_cutoff": relative_cutoff,
        "model_output": output_text,
        "extracted_answer": extracted_answer
    }

    return result


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        f"Loaded {len(data)} questions."
    )

    print(
        "Relative cutoff years:",
        list(RELATIVE_CUTOFF_YEARS)
    )

    # -----------------------------------------------------
    # Initialize Result Container
    #
    # Structure:
    #
    # model
    #   └── prompt
    #       └── relative_cutoff
    #           └── question results
    # -----------------------------------------------------

    file_results = {
        model: {
            prompt_style: {
                str(relative_cutoff): []
                for relative_cutoff
                in RELATIVE_CUTOFF_YEARS
            }
            for prompt_style
            in prompt_variants
        }
        for model
        in model_list
    }

    # -----------------------------------------------------
    # Run Experiments
    # -----------------------------------------------------

    for model, api_type in model_list.items():

        for prompt_style, prompt_template in (
            prompt_variants.items()
        ):

            for relative_cutoff in (
                RELATIVE_CUTOFF_YEARS
            ):

                print()
                print("=" * 70)

                print(
                    f"Model: {model}"
                )

                print(
                    f"Prompt: {prompt_style}"
                )

                print(
                    f"Relative cutoff: "
                    f"{relative_cutoff}"
                )

                print("=" * 70)

                # Create a unique intermediate file
                # for each model, prompt, and cutoff.
                safe_model_name = (
                    model.replace("/", "_")
                )

                inter_path = os.path.join(
                    intermediate_dir,
                    (
                        f"{safe_model_name}"
                        f"__{prompt_style}"
                        f"__relative_{relative_cutoff}"
                        f".jsonl"
                    )
                )

                # Remove an existing intermediate file
                # so that rerunning the same condition
                # does not append duplicate results.
                if os.path.exists(
                    inter_path
                ):

                    os.remove(
                        inter_path
                    )

                futures = []

                # -------------------------------------------------
                # Parallelize across questions.
                # Steps within one question remain sequential.
                # -------------------------------------------------

                with ThreadPoolExecutor(
                    max_workers=MAX_WORKERS
                ) as ex:

                    for pair in data:

                        futures.append(
                            ex.submit(
                                process_one,
                                pair,
                                model,
                                api_type,
                                prompt_style,
                                prompt_template,
                                relative_cutoff,
                                inter_path
                            )
                        )

                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=(
                            f"{prompt_style} "
                            f"({relative_cutoff})"
                        )
                    ):

                        try:

                            res = fut.result()

                            file_results[
                                model
                            ][
                                prompt_style
                            ][
                                str(relative_cutoff)
                            ].append(
                                res
                            )

                        except Exception as e:

                            print(
                                f"[ERROR] "
                                f"{model} - "
                                f"{prompt_style} - "
                                f"{relative_cutoff}: "
                                f"{e}"
                            )

                # -------------------------------------------------
                # Save the complete results after
                # each experimental condition.
                # This protects against losing all results
                # if a later condition fails.
                # -------------------------------------------------

                results_path = os.path.join(
                    output_path,
                    "historical_results.json"
                )

                with open(
                    results_path,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        file_results,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                print(
                    f"Results saved to: {results_path}"
                )

    print()
    print("=" * 70)
    print("All experiments completed.")
    print("=" * 70)