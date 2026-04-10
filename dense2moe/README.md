## Conversion Workflow

1. Perform distributed inference on calibration datasets, capturing per-layer activations and final-layer attention scores.

   ```python
   bash forward_calibration.sh
   ```

   This step generates a timestamped forward file in `output/`, for example:
   `Qwen3-Embedding-0.6B_domain_20260410_174520_forward.pt`.

2. Parsing distributed inference results and computing TokenScore.

   ```python
   python parse_forward_out.py \
     --forward_data_file /root/output/Qwen3-Embedding-0.6B_domain_20260410_174520_forward.pt
   ```

   By default, `parse_forward_out.py` enables uniform token score mode
   (`token_score = 1` for all tokens). To use the original SCS + RSS scoring, add:

   ```python
   --no_uniform_token_score
   ```

   It will save:
   - `output/token_scores/{forward_basename}_token_scores.pt`
   - `output/activations/{forward_basename}_activations_layer{layer}.pt`

3. Compute neuron domain score vectors layer-by-layer based on domain samples and the TokenScore-weighted importance formula.

   ```python
   python calculate_neuron_scores.py \
     --token_score_file /root/output/token_scores/Qwen3-Embedding-0.6B_domain_20260410_174520_forward_token_scores.pt
   ```

   `calculate_neuron_scores.py` will auto-infer the same run id from `token_score_file`,
   then load matching activation files and write outputs with the same suffix.

4. Layer-wise neuron clustering (using domain score vectors as distance) to assign cluster_ids (a list of expert indices per layer). For each split expert, domain coverage is calculated: a domain dimension scores 1 if the average neuron score in that sub-domain exceeds the top 50%, and 0 otherwise.

   ```python
   python split_expert_mlp.py \
     --run_id Qwen3-Embedding-0.6B_domain_20260410_174520_forward \
     --k 8
   ```

   It will save:
   - `output/expert_splits/neuron_splits_layer{layer}_k8_{run_id}.json`

5. Converting Qwen3 MLP to Den2MoEESparseMoeBlock layer-by-layer: initialization of shared/routed/zero-experts and router parameters.

   ```python
   python convert_den2moee.py \
     --time_stamp Qwen3-Embedding-0.6B_domain_20260410_174520_forward \
     --num_experts 8 \
     --top_k 2
   ```


