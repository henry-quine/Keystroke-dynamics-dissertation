from pathlib import Path

import pandas as pd

import main_pipeline as pipeline


# =============================================================================
# RUN THE FINAL DISSERTATION EXPERIMENTS
# =============================================================================
#
# This script runs the 9 final dissertation experiments.
#
# The experiments answer four research questions:
#
# 1. Does authentication remain reliable across three later sessions?
# 2. Does time pressure or cognitive distraction affect authentication?
# 3. Does using more enrollment sessions improve later authentication?
# 4. Does using more enrollment sessions improve later authentication?
#
# Each experiment first tunes the model settings and then trains the final
# models. The testing sessions are later sessions that were not used to train
# the final models.
#
# The data is prepared once. All experiments reuse the same feature tables, so
# feature extraction is not repeated 9 times.


THIS_FOLDER = Path(__file__).resolve().parent

# Keep the definitive results separate from earlier exploratory runs.
OUTPUT_FOLDER = THIS_FOLDER / "dissertation_outputs"


def main():
    # Store the results from every experiment so they can be joined at the end.
    all_results = []

    # Read the clean CSV and calculate the shared feature tables once.
    prepared_data = pipeline.prepare_data_once(OUTPUT_FOLDER)

    # =========================================================================
    # PART 1: ROBUSTNESS
    # =========================================================================
    # Train and test on the same type of typing to measure robustness across sessions.

    # =========================================================================
    # EXPERIMENT 1: FIXED TEXT ACROSS LATER SESSIONS
    # =========================================================================
    # Train on normal fixed text from Sessions 1 and 2, then test Sessions 3 to
    # 5 separately. This is the main controlled benchmark.

    experiment_results = pipeline.run_experiment(
        experiment_name="fixed_normal_across_sessions",

        prepared_data=prepared_data,

        training_data_type="fixed normal",
        testing_data_type="fixed normal",

        tuning_training_sessions=[1],
        tuning_validation_session=2,

        final_training_sessions=[1, 2],
        final_test_sessions=[3, 4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # EXPERIMENT 2: SEMI-FIXED TEXT ACROSS LATER SESSIONS
    # =========================================================================
    # Train on normal semi-fixed text from Sessions 1 and 2, then test Sessions
    # 3 to 5 separately.

    experiment_results = pipeline.run_experiment(
        experiment_name="semi_fixed_normal_across_sessions",

        prepared_data=prepared_data,

        training_data_type="semi-fixed normal",
        testing_data_type="semi-fixed normal",

        tuning_training_sessions=[1],
        tuning_validation_session=2,

        final_training_sessions=[1, 2],
        final_test_sessions=[3, 4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # EXPERIMENT 3: FREE TEXT ACROSS LATER SESSIONS
    # =========================================================================
    # Train on free-text windows from Sessions 1 and 2, then test Sessions 3 to
    # 5 separately. This is the closest test to natural continuous typing.

    experiment_results = pipeline.run_experiment(
        experiment_name="free_normal_across_sessions",

        prepared_data=prepared_data,

        training_data_type="free normal",
        testing_data_type="free normal",

        tuning_training_sessions=[1],
        tuning_validation_session=2,

        final_training_sessions=[1, 2],
        final_test_sessions=[3, 4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # PART 2: HUMAN VARIABILITY CONDITIONS
    # =========================================================================
    # Train on normal fixed text, then change the condition used for testing.

    # =========================================================================
    # EXPERIMENT 4: NORMAL FIXED ENROLLMENT TESTED ON RUSHED TYPING
    # =========================================================================
    # Test rushed fixed text from Sessions 3 to 5. Compare with Experiment 1 to
    # measure the effect of time pressure.

    experiment_results = pipeline.run_experiment(
        experiment_name="fixed_normal_to_rushed",

        prepared_data=prepared_data,

        training_data_type="fixed normal",
        testing_data_type="fixed rushed",

        tuning_training_sessions=[1],
        tuning_validation_session=2,

        final_training_sessions=[1, 2],
        final_test_sessions=[3, 4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # EXPERIMENT 5: NORMAL FIXED ENROLLMENT TESTED ON DISTRACTED TYPING
    # =========================================================================
    # Test distracted fixed text from Sessions 3 to 5. Compare with Experiment
    # 1 to measure the effect of cognitive distraction.

    experiment_results = pipeline.run_experiment(
        experiment_name="fixed_normal_to_distracted",

        prepared_data=prepared_data,

        training_data_type="fixed normal",
        testing_data_type="fixed distracted",

        tuning_training_sessions=[1],
        tuning_validation_session=2,

        final_training_sessions=[1, 2],
        final_test_sessions=[3, 4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # PART 3: ENROLLMENT SIZE AND MODEL UPDATING
    # =========================================================================
    # Change the number of enrollment sessions and always test on later data.

    # -------------------------------------------------------------------------
    # Fixed-text enrollment
    # -------------------------------------------------------------------------

    # =========================================================================
    # EXPERIMENT 6: UPDATE FIXED ENROLLMENT WITH SESSION 3
    # =========================================================================
    # Train the final models on Sessions 1 to 3 and test Sessions 4 and 5.

    experiment_results = pipeline.run_experiment(
        experiment_name="fixed_normal_updated_through_session_3",

        prepared_data=prepared_data,

        training_data_type="fixed normal",
        testing_data_type="fixed normal",

        tuning_training_sessions=[1, 2],
        tuning_validation_session=3,

        final_training_sessions=[1, 2, 3],
        final_test_sessions=[4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # EXPERIMENT 7: UPDATE FIXED ENROLLMENT WITH SESSION 4
    # =========================================================================
    # Train the final models on Sessions 1 to 4 and test Session 5.

    experiment_results = pipeline.run_experiment(
        experiment_name="fixed_normal_updated_through_session_4",

        prepared_data=prepared_data,

        training_data_type="fixed normal",
        testing_data_type="fixed normal",

        tuning_training_sessions=[1, 2, 3],
        tuning_validation_session=4,

        final_training_sessions=[1, 2, 3, 4],
        final_test_sessions=[5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # -------------------------------------------------------------------------
    # Free-text enrollment
    # -------------------------------------------------------------------------

    # =========================================================================
    # EXPERIMENT 8: UPDATE FREE-TEXT ENROLLMENT WITH SESSION 3
    # =========================================================================
    # Train the final models on Sessions 1 to 3 and test Sessions 4 and 5.
    # Compare with Experiment 3 on the same free-text windows.

    experiment_results = pipeline.run_experiment(
        experiment_name="free_normal_updated_through_session_3",

        prepared_data=prepared_data,

        training_data_type="free normal",
        testing_data_type="free normal",

        tuning_training_sessions=[1, 2],
        tuning_validation_session=3,

        final_training_sessions=[1, 2, 3],
        final_test_sessions=[4, 5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # =========================================================================
    # EXPERIMENT 9: UPDATE FREE-TEXT ENROLLMENT WITH SESSION 4
    # =========================================================================
    # Train the final models on Sessions 1 to 4 and test Session 5. Compare
    # Experiments 3 and 8 on the same free-text windows.

    experiment_results = pipeline.run_experiment(
        experiment_name="free_normal_updated_through_session_4",

        prepared_data=prepared_data,

        training_data_type="free normal",
        testing_data_type="free normal",

        tuning_training_sessions=[1, 2, 3],
        tuning_validation_session=4,

        final_training_sessions=[1, 2, 3, 4],
        final_test_sessions=[5],

        output_folder=OUTPUT_FOLDER,
    )
    all_results.append(experiment_results)

    # Join the session results and save one final comparison CSV file.
    comparison_table = pd.concat(all_results, ignore_index=True)
    comparison_file = OUTPUT_FOLDER / "experiment_comparison.csv"
    comparison_table.to_csv(comparison_file, index=False)

    print("")
    print("=" * 70)
    print("ALL DISSERTATION EXPERIMENTS DONE")
    print("=" * 70)
    print("Experiments completed:", len(all_results))
    print("Comparison file:")
    print(comparison_file)


if __name__ == "__main__":
    main()
