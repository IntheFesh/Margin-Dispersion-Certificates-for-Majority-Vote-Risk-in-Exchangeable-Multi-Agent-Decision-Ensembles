from __future__ import annotations
import numpy as np
import pandas as pd


def sample_protocol_A1_agents(num_samples, template_pool, seed) -> pd.DataFrame:
    if not template_pool:
        raise ValueError("template_pool cannot be empty")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sample_id": range(num_samples), "prompt_template_id": rng.choice(template_pool, size=num_samples), "decoding_seed": rng.integers(0, 2**31 - 1, size=num_samples)})


def sample_protocol_A2_agents(num_samples, model_variants, seed) -> pd.DataFrame:
    if not model_variants:
        raise ValueError("model_variants cannot be empty")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sample_id": range(num_samples), "model_id": rng.choice(model_variants, size=num_samples), "decoding_seed": rng.integers(0, 2**31 - 1, size=num_samples)})


def build_panel_a_sampling_plan(instances, protocol_config, K_ref, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plans = []
    for instance_id in instances:
        proto = protocol_config["protocol"]
        if proto == "A1":
            sampled = sample_protocol_A1_agents(K_ref, protocol_config["template_pool"], int(rng.integers(0, 2**31 - 1)))
            sampled["model_id"] = protocol_config["base_model"]
        elif proto == "A2":
            sampled = sample_protocol_A2_agents(K_ref, protocol_config["model_variants"], int(rng.integers(0, 2**31 - 1)))
            sampled["prompt_template_id"] = protocol_config.get("prompt_template_id", "default")
        else:
            raise ValueError(f"Unsupported protocol: {proto}")
        sampled["instance_id"] = instance_id
        sampled["protocol"] = proto
        sampled["temperature"] = protocol_config["temperature"]
        plans.append(sampled[["instance_id", "sample_id", "protocol", "model_id", "prompt_template_id", "temperature", "decoding_seed"]])
    return pd.concat(plans, ignore_index=True)
