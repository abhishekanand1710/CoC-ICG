cd arcade-nl2code
DATASET_ROOT=arcade_nl2code/annotated_dataset/dataset
MAX_PROMPT_SIZE=900
faketime "2022-12-10 12:00:00" python -m arcade_nl2code.annotated_dataset.generate_schema_augmented_prompts \
    --dataset ${DATASET_ROOT}/existing_tasks/dataset.json \
    --output_folder ${DATASET_ROOT}/existing_tasks/derived_datasets/ \
    --runtime_artifacts_root ${DATASET_ROOT}/existing_tasks/artifacts/ \
    --schema_representation_method "originating_dfs.header_description.after_variable_cell" \
    --max_prompt_size ${MAX_PROMPT_SIZE} \
    --truncate_metadata_path ${DATASET_ROOT}/existing_tasks/derived_datasets/dataset.schema.originating_dfs.header_description.after_variable_cell.maxp900.maxp_no_prefix-1.maxctxcell-1.truncate_metadata.json