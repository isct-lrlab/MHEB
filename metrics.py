
import os
import json
import csv


# =========================================================
# Configuration
# =========================================================

output_path = "results"

results_path = os.path.join(
    output_path,
    "historical_results.json"
)

summary_dir = os.path.join(
    output_path,
    "summary"
)

# Relative cutoff years to summarize.
RELATIVE_CUTOFF_YEARS = range(-10, 1)


# =========================================================
# Utility Functions
# =========================================================

def calculate_accuracy(results):
    """
    Calculate accuracy using the expected answer
    stored in "answer_after_cutoff".

    Invalid outputs such as "N/A" are counted as incorrect.
    """

    if not results:
        return 0.0

    correct = 0

    for result in results:

        prediction = result["extracted_answer"]
        gold = result["answer_after_cutoff"]

        if (
            prediction != "N/A"
            and prediction == gold
        ):
            correct += 1

    return correct / len(results)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Load Results
    # -----------------------------------------------------

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    with open(
        results_path,
        "r",
        encoding="utf-8"
    ) as f:

        file_results = json.load(f)

    print(
        f"Loaded results from: {results_path}"
    )

    # -----------------------------------------------------
    # Create Summary Directory
    # -----------------------------------------------------

    os.makedirs(
        summary_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Create Summary Tables
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Creating summary tables...")
    print("=" * 70)

    cutoff_list = list(
        RELATIVE_CUTOFF_YEARS
    )

    # -----------------------------------------------------
    # Process Each Model
    # -----------------------------------------------------

    for model in file_results:

        safe_model_name = (
            model.replace("/", "_")
        )

        summary = {}

        # -------------------------------------------------
        # Process Each Prompt
        # -------------------------------------------------

        for prompt_style in file_results[model]:

            summary[prompt_style] = {}

            accuracies = []

            # ---------------------------------------------
            # Calculate Accuracy for Each Cutoff
            # ---------------------------------------------

            for relative_cutoff in cutoff_list:

                results = file_results[
                    model
                ][
                    prompt_style
                ][
                    str(relative_cutoff)
                ]

                accuracy = calculate_accuracy(
                    results
                )

                summary[
                    prompt_style
                ][
                    relative_cutoff
                ] = accuracy

                accuracies.append(
                    accuracy
                )

            # ---------------------------------------------
            # Calculate Average Accuracy
            # ---------------------------------------------

            if accuracies:

                summary[
                    prompt_style
                ]["avg"] = (
                    sum(accuracies)
                    / len(accuracies)
                )

            else:

                summary[
                    prompt_style
                ]["avg"] = 0.0

        # =================================================
        # Save Summary as CSV
        # =================================================

        csv_path = os.path.join(
            summary_dir,
            f"{safe_model_name}_accuracy.csv"
        )

        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(f)

            header = (
                ["Prompt"]
                + [
                    str(x)
                    for x in cutoff_list
                ]
                + ["avg"]
            )

            writer.writerow(
                header
            )

            for prompt_style in summary:

                row = [
                    prompt_style
                ]

                for relative_cutoff in (
                    cutoff_list
                ):

                    row.append(
                        f"{summary[prompt_style][relative_cutoff]:.3f}"
                    )

                row.append(
                    f"{summary[prompt_style]['avg']:.3f}"
                )

                writer.writerow(
                    row
                )

        # =================================================
        # Save Summary as JSON
        # =================================================

        json_path = os.path.join(
            summary_dir,
            f"{safe_model_name}_accuracy.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2
            )

        # =================================================
        # Print Summary Table
        # =================================================

        print()
        print(
            f"Model: {model}"
        )

        print()

        header = (
            ["Prompt"]
            + [
                str(x)
                for x in cutoff_list
            ]
            + ["avg"]
        )

        print(
            "\t".join(header)
        )

        for prompt_style in summary:

            row = [
                prompt_style
            ]

            for relative_cutoff in (
                cutoff_list
            ):

                row.append(
                    f"{summary[prompt_style][relative_cutoff]:.3f}"
                )

            row.append(
                f"{summary[prompt_style]['avg']:.3f}"
            )

            print(
                "\t".join(row)
            )

    # =====================================================
    # Finish
    # =====================================================

    print()
    print("=" * 70)
    print("Summary tables created successfully.")
    print("=" * 70)
