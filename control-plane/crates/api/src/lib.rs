use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobMode {
    Sync,
    Async,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainBackend {
    Fsdp,
    Megatron,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourcePlan {
    pub actor_num_nodes: u32,
    pub actor_num_gpus_per_node: u32,
    pub critic_num_nodes: Option<u32>,
    pub critic_num_gpus_per_node: Option<u32>,
    pub rollout_num_gpus: u32,
    pub rollout_num_gpus_per_engine: u32,
    pub num_gpus_per_node: u32,
    pub colocate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetConfig {
    pub prompt_data: String,
    pub input_key: String,
    pub label_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutConfig {
    pub batch_size: u32,
    pub n_samples_per_prompt: u32,
    pub max_response_len: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainConfig {
    pub global_batch_size: u32,
    pub num_rollout: u32,
    pub update_weights_interval: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalConfig {
    pub eval_interval: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaveConfig {
    pub save_interval: Option<u32>,
    pub output_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaultToleranceConfig {
    pub enabled: bool,
    pub max_retries: u32,
    pub retry_backoff_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobSpec {
    pub job_name: String,
    pub mode: JobMode,
    pub backend: TrainBackend,
    pub resource_plan: ResourcePlan,
    pub dataset_config: DatasetConfig,
    pub rollout_config: RolloutConfig,
    pub train_config: TrainConfig,
    pub eval_config: EvalConfig,
    pub save_config: SaveConfig,
    pub fault_tolerance_config: FaultToleranceConfig,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("job_name must not be empty")]
    EmptyJobName,
    #[error("actor_num_nodes and actor_num_gpus_per_node must be > 0")]
    InvalidActorResources,
    #[error("rollout_num_gpus and rollout_num_gpus_per_engine must be > 0")]
    InvalidRolloutResources,
    #[error("rollout_num_gpus must be divisible by rollout_num_gpus_per_engine")]
    InvalidRolloutPartition,
    #[error("global_batch_size, num_rollout, and update_weights_interval must be > 0")]
    InvalidTrainConfig,
    #[error("rollout batch_size and n_samples_per_prompt must be > 0")]
    InvalidRolloutConfig,
    #[error("prompt_data must not be empty")]
    EmptyPromptData,
    #[error(
        "in colocate mode, rollout_num_gpus must equal actor_num_nodes * actor_num_gpus_per_node"
    )]
    ColocateRolloutMismatch,
}

impl JobSpec {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.job_name.trim().is_empty() {
            return Err(ValidationError::EmptyJobName);
        }

        if self.resource_plan.actor_num_nodes == 0
            || self.resource_plan.actor_num_gpus_per_node == 0
        {
            return Err(ValidationError::InvalidActorResources);
        }

        if self.resource_plan.rollout_num_gpus == 0
            || self.resource_plan.rollout_num_gpus_per_engine == 0
        {
            return Err(ValidationError::InvalidRolloutResources);
        }

        if self.resource_plan.rollout_num_gpus % self.resource_plan.rollout_num_gpus_per_engine != 0
        {
            return Err(ValidationError::InvalidRolloutPartition);
        }

        if self.rollout_config.batch_size == 0 || self.rollout_config.n_samples_per_prompt == 0 {
            return Err(ValidationError::InvalidRolloutConfig);
        }

        if self.train_config.global_batch_size == 0
            || self.train_config.num_rollout == 0
            || self.train_config.update_weights_interval == 0
        {
            return Err(ValidationError::InvalidTrainConfig);
        }

        if self.dataset_config.prompt_data.trim().is_empty() {
            return Err(ValidationError::EmptyPromptData);
        }

        if self.resource_plan.colocate {
            let actor_gpus =
                self.resource_plan.actor_num_nodes * self.resource_plan.actor_num_gpus_per_node;
            if self.resource_plan.rollout_num_gpus != actor_gpus {
                return Err(ValidationError::ColocateRolloutMismatch);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Created,
    Starting,
    Running,
    Paused,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoopPhase {
    Init,
    PrepareRollout,
    Generate,
    Train,
    UpdateWeights,
    Eval,
    Save,
    Cleanup,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RolloutCursor {
    pub current_rollout_id: u32,
    pub total_rollouts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobRuntime {
    pub job_id: Uuid,
    pub state: JobState,
    pub phase: LoopPhase,
    pub cursor: RolloutCursor,
    pub fence_token: u64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobEvent {
    pub job_id: Uuid,
    pub phase: LoopPhase,
    pub state: JobState,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn valid_spec() -> JobSpec {
        JobSpec {
            job_name: "smoke".to_string(),
            mode: JobMode::Sync,
            backend: TrainBackend::Fsdp,
            resource_plan: ResourcePlan {
                actor_num_nodes: 1,
                actor_num_gpus_per_node: 2,
                critic_num_nodes: None,
                critic_num_gpus_per_node: None,
                rollout_num_gpus: 2,
                rollout_num_gpus_per_engine: 1,
                num_gpus_per_node: 8,
                colocate: false,
            },
            dataset_config: DatasetConfig {
                prompt_data: "/tmp/prompts.jsonl".to_string(),
                input_key: "prompt".to_string(),
                label_key: "label".to_string(),
            },
            rollout_config: RolloutConfig {
                batch_size: 16,
                n_samples_per_prompt: 8,
                max_response_len: Some(2048),
            },
            train_config: TrainConfig {
                global_batch_size: 128,
                num_rollout: 100,
                update_weights_interval: 1,
            },
            eval_config: EvalConfig {
                eval_interval: Some(10),
            },
            save_config: SaveConfig {
                save_interval: Some(20),
                output_dir: Some("/tmp/out".to_string()),
            },
            fault_tolerance_config: FaultToleranceConfig {
                enabled: true,
                max_retries: 3,
                retry_backoff_ms: 250,
            },
        }
    }

    #[test]
    fn valid_spec_passes() {
        let spec = valid_spec();
        assert_eq!(spec.validate(), Ok(()));
    }

    #[test]
    fn empty_job_name_fails() {
        let mut spec = valid_spec();
        spec.job_name = "   ".to_string();
        assert_eq!(spec.validate(), Err(ValidationError::EmptyJobName));
    }

    #[test]
    fn zero_actor_resources_fail() {
        let mut spec = valid_spec();
        spec.resource_plan.actor_num_nodes = 0;
        assert_eq!(spec.validate(), Err(ValidationError::InvalidActorResources));
    }

    #[test]
    fn zero_rollout_resources_fail() {
        let mut spec = valid_spec();
        spec.resource_plan.rollout_num_gpus_per_engine = 0;
        assert_eq!(
            spec.validate(),
            Err(ValidationError::InvalidRolloutResources)
        );
    }

    #[test]
    fn invalid_rollout_partition_fails() {
        let mut spec = valid_spec();
        spec.resource_plan.rollout_num_gpus = 3;
        spec.resource_plan.rollout_num_gpus_per_engine = 2;
        assert_eq!(
            spec.validate(),
            Err(ValidationError::InvalidRolloutPartition)
        );
    }

    #[test]
    fn invalid_rollout_config_fails() {
        let mut spec = valid_spec();
        spec.rollout_config.n_samples_per_prompt = 0;
        assert_eq!(spec.validate(), Err(ValidationError::InvalidRolloutConfig));
    }

    #[test]
    fn invalid_train_config_fails() {
        let mut spec = valid_spec();
        spec.train_config.update_weights_interval = 0;
        assert_eq!(spec.validate(), Err(ValidationError::InvalidTrainConfig));
    }

    #[test]
    fn empty_prompt_data_fails() {
        let mut spec = valid_spec();
        spec.dataset_config.prompt_data = " ".to_string();
        assert_eq!(spec.validate(), Err(ValidationError::EmptyPromptData));
    }

    #[test]
    fn colocate_requires_matching_rollout_gpu_count() {
        let mut spec = valid_spec();
        spec.resource_plan.colocate = true;
        spec.resource_plan.rollout_num_gpus = 1;
        assert_eq!(
            spec.validate(),
            Err(ValidationError::ColocateRolloutMismatch)
        );
    }

    #[test]
    fn colocate_with_matching_rollout_gpu_count_passes() {
        let mut spec = valid_spec();
        spec.resource_plan.colocate = true;
        spec.resource_plan.rollout_num_gpus =
            spec.resource_plan.actor_num_nodes * spec.resource_plan.actor_num_gpus_per_node;
        assert_eq!(spec.validate(), Ok(()));
    }

    #[test]
    fn enum_serialization_uses_snake_case() {
        let mode = serde_json::to_value(JobMode::Async).expect("serialize mode");
        let backend = serde_json::to_value(TrainBackend::Megatron).expect("serialize backend");
        let state = serde_json::to_value(JobState::Running).expect("serialize state");
        let phase = serde_json::to_value(LoopPhase::UpdateWeights).expect("serialize phase");
        assert_eq!(mode, json!("async"));
        assert_eq!(backend, json!("megatron"));
        assert_eq!(state, json!("running"));
        assert_eq!(phase, json!("update_weights"));
    }

    #[test]
    fn jobspec_roundtrip_json() {
        let spec = valid_spec();
        let encoded = serde_json::to_string(&spec).expect("serialize jobspec");
        let decoded: JobSpec = serde_json::from_str(&encoded).expect("deserialize jobspec");
        assert_eq!(decoded, spec);
    }

    #[test]
    fn runtime_and_event_roundtrip_json() {
        let runtime = JobRuntime {
            job_id: Uuid::new_v4(),
            state: JobState::Running,
            phase: LoopPhase::Train,
            cursor: RolloutCursor {
                current_rollout_id: 7,
                total_rollouts: 20,
            },
            fence_token: 11,
            updated_at: Utc::now(),
        };
        let event = JobEvent {
            job_id: runtime.job_id,
            phase: runtime.phase,
            state: runtime.state,
            message: "progress".to_string(),
            timestamp: Utc::now(),
        };

        let runtime_val: Value = serde_json::to_value(&runtime).expect("serialize runtime");
        let runtime_back: JobRuntime =
            serde_json::from_value(runtime_val).expect("deserialize runtime");
        assert_eq!(runtime_back, runtime);

        let event_val: Value = serde_json::to_value(&event).expect("serialize event");
        let event_back: JobEvent = serde_json::from_value(event_val).expect("deserialize event");
        assert_eq!(event_back, event);
    }
}
