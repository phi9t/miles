use async_trait::async_trait;
use chrono::Utc;
use miles_control_api::{JobRuntime, JobSpec};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StoreError {
    #[error("job {0} not found")]
    NotFound(Uuid),
    #[error("io error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("invalid snapshot filename: {0}")]
    InvalidSnapshotName(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredJob {
    pub spec: JobSpec,
    pub runtime: JobRuntime,
}

#[async_trait]
pub trait StateStore: Send + Sync {
    async fn put_job(&self, spec: JobSpec, runtime: JobRuntime) -> Result<(), StoreError>;
    async fn get_job(&self, job_id: Uuid) -> Result<StoredJob, StoreError>;
    async fn update_runtime(&self, runtime: JobRuntime) -> Result<(), StoreError>;
    async fn list_jobs(&self) -> Result<Vec<StoredJob>, StoreError>;
    async fn snapshot_job(&self, job_id: Uuid) -> Result<(), StoreError>;
    async fn load_snapshot(&self, job_id: Uuid) -> Result<StoredJob, StoreError>;
    async fn list_snapshots(&self) -> Result<Vec<Uuid>, StoreError>;
}

#[derive(Debug, Default, Clone)]
pub struct InMemoryStateStore {
    jobs: Arc<RwLock<HashMap<Uuid, StoredJob>>>,
}

#[async_trait]
impl StateStore for InMemoryStateStore {
    async fn put_job(&self, spec: JobSpec, runtime: JobRuntime) -> Result<(), StoreError> {
        self.jobs
            .write()
            .await
            .insert(runtime.job_id, StoredJob { spec, runtime });
        Ok(())
    }

    async fn get_job(&self, job_id: Uuid) -> Result<StoredJob, StoreError> {
        self.jobs
            .read()
            .await
            .get(&job_id)
            .cloned()
            .ok_or(StoreError::NotFound(job_id))
    }

    async fn update_runtime(&self, mut runtime: JobRuntime) -> Result<(), StoreError> {
        runtime.updated_at = Utc::now();
        let mut guard = self.jobs.write().await;
        let Some(stored) = guard.get_mut(&runtime.job_id) else {
            return Err(StoreError::NotFound(runtime.job_id));
        };
        stored.runtime = runtime;
        Ok(())
    }

    async fn list_jobs(&self) -> Result<Vec<StoredJob>, StoreError> {
        Ok(self.jobs.read().await.values().cloned().collect())
    }

    async fn snapshot_job(&self, job_id: Uuid) -> Result<(), StoreError> {
        let guard = self.jobs.read().await;
        if guard.contains_key(&job_id) {
            Ok(())
        } else {
            Err(StoreError::NotFound(job_id))
        }
    }

    async fn load_snapshot(&self, job_id: Uuid) -> Result<StoredJob, StoreError> {
        self.get_job(job_id).await
    }

    async fn list_snapshots(&self) -> Result<Vec<Uuid>, StoreError> {
        let mut ids = self.jobs.read().await.keys().copied().collect::<Vec<_>>();
        ids.sort();
        Ok(ids)
    }
}

#[derive(Debug, Clone)]
pub struct FileStateStore {
    jobs: Arc<RwLock<HashMap<Uuid, StoredJob>>>,
    snapshot_dir: Arc<PathBuf>,
}

impl FileStateStore {
    pub async fn new<P: AsRef<Path>>(snapshot_dir: P) -> Result<Self, StoreError> {
        let dir = snapshot_dir.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?;

        let store = Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            snapshot_dir: Arc::new(dir),
        };
        store.restore_all().await?;
        Ok(store)
    }

    pub fn snapshot_dir(&self) -> &Path {
        self.snapshot_dir.as_ref().as_path()
    }

    fn snapshot_path(&self, job_id: Uuid) -> PathBuf {
        self.snapshot_dir.join(format!("{job_id}.json"))
    }

    async fn persist_job(&self, job_id: Uuid) -> Result<(), StoreError> {
        let stored = self.get_job(job_id).await?;
        let encoded = serde_json::to_vec_pretty(&stored)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;
        let path = self.snapshot_path(job_id);
        let tmp_path = self.snapshot_dir.join(format!("{job_id}.tmp"));

        tokio::fs::write(&tmp_path, encoded)
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?;
        tokio::fs::rename(&tmp_path, &path)
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn restore_all(&self) -> Result<(), StoreError> {
        let mut entries = tokio::fs::read_dir(self.snapshot_dir())
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?;
        let mut loaded: Vec<(Uuid, StoredJob)> = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?
        {
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|v| v.to_str())
                .ok_or_else(|| StoreError::InvalidSnapshotName(path.display().to_string()))?;
            let job_id = Uuid::parse_str(stem)
                .map_err(|_| StoreError::InvalidSnapshotName(stem.to_string()))?;

            let data = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| StoreError::Io(e.to_string()))?;
            let stored = serde_json::from_str::<StoredJob>(&data)
                .map_err(|e| StoreError::Serialization(e.to_string()))?;

            if stored.runtime.job_id != job_id {
                return Err(StoreError::Serialization(format!(
                    "snapshot id mismatch: filename {job_id}, payload {}",
                    stored.runtime.job_id
                )));
            }
            loaded.push((job_id, stored));
        }

        let mut guard = self.jobs.write().await;
        for (job_id, stored) in loaded {
            guard.insert(job_id, stored);
        }
        Ok(())
    }
}

#[async_trait]
impl StateStore for FileStateStore {
    async fn put_job(&self, spec: JobSpec, runtime: JobRuntime) -> Result<(), StoreError> {
        let job_id = runtime.job_id;
        self.jobs
            .write()
            .await
            .insert(job_id, StoredJob { spec, runtime });
        self.persist_job(job_id).await
    }

    async fn get_job(&self, job_id: Uuid) -> Result<StoredJob, StoreError> {
        self.jobs
            .read()
            .await
            .get(&job_id)
            .cloned()
            .ok_or(StoreError::NotFound(job_id))
    }

    async fn update_runtime(&self, mut runtime: JobRuntime) -> Result<(), StoreError> {
        runtime.updated_at = Utc::now();
        {
            let mut guard = self.jobs.write().await;
            let Some(stored) = guard.get_mut(&runtime.job_id) else {
                return Err(StoreError::NotFound(runtime.job_id));
            };
            stored.runtime = runtime.clone();
        }
        self.persist_job(runtime.job_id).await
    }

    async fn list_jobs(&self) -> Result<Vec<StoredJob>, StoreError> {
        Ok(self.jobs.read().await.values().cloned().collect())
    }

    async fn snapshot_job(&self, job_id: Uuid) -> Result<(), StoreError> {
        self.persist_job(job_id).await
    }

    async fn load_snapshot(&self, job_id: Uuid) -> Result<StoredJob, StoreError> {
        let path = self.snapshot_path(job_id);
        if !path.exists() {
            return Err(StoreError::NotFound(job_id));
        }

        let data = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?;
        let stored = serde_json::from_str::<StoredJob>(&data)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;
        if stored.runtime.job_id != job_id {
            return Err(StoreError::Serialization(format!(
                "snapshot id mismatch: requested {job_id}, payload {}",
                stored.runtime.job_id
            )));
        }

        self.jobs.write().await.insert(job_id, stored.clone());
        Ok(stored)
    }

    async fn list_snapshots(&self) -> Result<Vec<Uuid>, StoreError> {
        let mut entries = tokio::fs::read_dir(self.snapshot_dir())
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?;
        let mut ids = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StoreError::Io(e.to_string()))?
        {
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|v| v.to_str())
                .ok_or_else(|| StoreError::InvalidSnapshotName(path.display().to_string()))?;
            let job_id = Uuid::parse_str(stem)
                .map_err(|_| StoreError::InvalidSnapshotName(stem.to_string()))?;
            ids.push(job_id);
        }

        ids.sort();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use miles_control_api::{
        DatasetConfig, EvalConfig, FaultToleranceConfig, JobMode, JobSpec, JobState, LoopPhase,
        ResourcePlan, RolloutConfig, RolloutCursor, SaveConfig, TrainBackend, TrainConfig,
    };

    fn sample_spec() -> JobSpec {
        JobSpec {
            job_name: "job".to_string(),
            mode: JobMode::Sync,
            backend: TrainBackend::Fsdp,
            resource_plan: ResourcePlan {
                actor_num_nodes: 1,
                actor_num_gpus_per_node: 1,
                critic_num_nodes: None,
                critic_num_gpus_per_node: None,
                rollout_num_gpus: 1,
                rollout_num_gpus_per_engine: 1,
                num_gpus_per_node: 8,
                colocate: false,
            },
            dataset_config: DatasetConfig {
                prompt_data: "x".to_string(),
                input_key: "prompt".to_string(),
                label_key: "label".to_string(),
            },
            rollout_config: RolloutConfig {
                batch_size: 1,
                n_samples_per_prompt: 1,
                max_response_len: None,
            },
            train_config: TrainConfig {
                global_batch_size: 1,
                num_rollout: 1,
                update_weights_interval: 1,
            },
            eval_config: EvalConfig {
                eval_interval: None,
            },
            save_config: SaveConfig {
                save_interval: None,
                output_dir: None,
            },
            fault_tolerance_config: FaultToleranceConfig {
                enabled: false,
                max_retries: 0,
                retry_backoff_ms: 0,
            },
        }
    }

    fn runtime(job_id: Uuid) -> JobRuntime {
        JobRuntime {
            job_id,
            state: JobState::Created,
            phase: LoopPhase::Init,
            cursor: RolloutCursor {
                current_rollout_id: 0,
                total_rollouts: 1,
            },
            fence_token: 0,
            updated_at: Utc::now(),
        }
    }

    fn temp_snapshot_dir() -> PathBuf {
        std::env::temp_dir().join(format!("miles-control-state-store-{}", Uuid::new_v4()))
    }

    #[tokio::test]
    async fn put_and_get_job() {
        let store = InMemoryStateStore::default();
        let job_id = Uuid::new_v4();

        store
            .put_job(sample_spec(), runtime(job_id))
            .await
            .expect("put should succeed");
        let stored = store.get_job(job_id).await.expect("job should exist");
        assert_eq!(stored.runtime.job_id, job_id);
        assert_eq!(stored.runtime.state, JobState::Created);
    }

    #[tokio::test]
    async fn get_unknown_job_returns_not_found() {
        let store = InMemoryStateStore::default();
        let missing = Uuid::new_v4();
        let err = store
            .get_job(missing)
            .await
            .expect_err("missing job should fail");
        assert_eq!(err, StoreError::NotFound(missing));
    }

    #[tokio::test]
    async fn update_unknown_job_returns_not_found() {
        let store = InMemoryStateStore::default();
        let missing = Uuid::new_v4();
        let mut changed_runtime = runtime(missing);
        changed_runtime.state = JobState::Running;
        changed_runtime.phase = LoopPhase::Train;
        changed_runtime.fence_token = 3;

        let err = store
            .update_runtime(changed_runtime)
            .await
            .expect_err("missing update should fail");
        assert_eq!(err, StoreError::NotFound(missing));
    }

    #[tokio::test]
    async fn put_with_same_job_id_overwrites_previous_value() {
        let store = InMemoryStateStore::default();
        let job_id = Uuid::new_v4();
        store
            .put_job(sample_spec(), runtime(job_id))
            .await
            .expect("first put succeeds");

        let mut updated = runtime(job_id);
        updated.state = JobState::Running;
        updated.phase = LoopPhase::Train;
        updated.fence_token = 8;

        store
            .put_job(sample_spec(), updated.clone())
            .await
            .expect("overwrite put succeeds");

        let stored = store.get_job(job_id).await.expect("job should exist");
        assert_eq!(stored.runtime.state, JobState::Running);
        assert_eq!(stored.runtime.phase, LoopPhase::Train);
        assert_eq!(stored.runtime.fence_token, 8);
    }

    #[tokio::test]
    async fn update_runtime_changes_state_phase_and_refreshes_timestamp() {
        let store = InMemoryStateStore::default();
        let job_id = Uuid::new_v4();
        let mut initial = runtime(job_id);
        let before_update = initial.updated_at;

        store
            .put_job(sample_spec(), initial.clone())
            .await
            .expect("put succeeds");

        initial.state = JobState::Running;
        initial.phase = LoopPhase::Generate;
        initial.fence_token = 4;

        store
            .update_runtime(initial.clone())
            .await
            .expect("update succeeds");

        let stored = store.get_job(job_id).await.expect("job should exist");
        assert_eq!(stored.runtime.state, JobState::Running);
        assert_eq!(stored.runtime.phase, LoopPhase::Generate);
        assert_eq!(stored.runtime.fence_token, 4);
        assert!(stored.runtime.updated_at >= before_update);
    }

    #[tokio::test]
    async fn list_jobs_returns_all_inserted_jobs() {
        let store = InMemoryStateStore::default();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        for id in [id1, id2] {
            store
                .put_job(sample_spec(), runtime(id))
                .await
                .expect("put succeeds");
        }

        let jobs = store.list_jobs().await.expect("list succeeds");
        assert_eq!(jobs.len(), 2);
        let ids = jobs
            .into_iter()
            .map(|j| j.runtime.job_id)
            .collect::<Vec<_>>();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[tokio::test]
    async fn concurrent_puts_and_reads_are_safe() {
        let store = InMemoryStateStore::default();
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        let store_a = store.clone();
        let write_a = async move {
            store_a
                .put_job(sample_spec(), runtime(id_a))
                .await
                .expect("put A succeeds");
        };

        let store_b = store.clone();
        let write_b = async move {
            store_b
                .put_job(sample_spec(), runtime(id_b))
                .await
                .expect("put B succeeds");
        };

        tokio::join!(write_a, write_b);

        let (a, b) = tokio::join!(store.get_job(id_a), store.get_job(id_b));
        assert!(a.is_ok());
        assert!(b.is_ok());
    }

    #[tokio::test]
    async fn file_state_store_persists_and_restores_job() {
        let dir = temp_snapshot_dir();
        let store = FileStateStore::new(&dir)
            .await
            .expect("create file state store");
        let job_id = Uuid::new_v4();

        let mut job_runtime = runtime(job_id);
        job_runtime.state = JobState::Running;
        job_runtime.phase = LoopPhase::Train;
        job_runtime.fence_token = 11;

        store
            .put_job(sample_spec(), job_runtime.clone())
            .await
            .expect("persisted put");

        let restored = FileStateStore::new(&dir)
            .await
            .expect("restore file state store");
        let loaded = restored.get_job(job_id).await.expect("job loaded");
        assert_eq!(loaded.runtime.state, JobState::Running);
        assert_eq!(loaded.runtime.phase, LoopPhase::Train);
        assert_eq!(loaded.runtime.fence_token, 11);

        tokio::fs::remove_dir_all(&dir)
            .await
            .expect("cleanup temp dir");
    }

    #[tokio::test]
    async fn file_load_snapshot_missing_returns_not_found() {
        let dir = temp_snapshot_dir();
        let store = FileStateStore::new(&dir)
            .await
            .expect("create file state store");
        let missing = Uuid::new_v4();

        let err = store
            .load_snapshot(missing)
            .await
            .expect_err("missing snapshot should fail");
        assert_eq!(err, StoreError::NotFound(missing));

        tokio::fs::remove_dir_all(&dir)
            .await
            .expect("cleanup temp dir");
    }

    #[tokio::test]
    async fn file_list_snapshots_returns_all_job_ids() {
        let dir = temp_snapshot_dir();
        let store = FileStateStore::new(&dir)
            .await
            .expect("create file state store");
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        store
            .put_job(sample_spec(), runtime(id1))
            .await
            .expect("persist id1");
        store
            .put_job(sample_spec(), runtime(id2))
            .await
            .expect("persist id2");

        let ids = store.list_snapshots().await.expect("list snapshots");
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));

        tokio::fs::remove_dir_all(&dir)
            .await
            .expect("cleanup temp dir");
    }

    #[tokio::test]
    async fn file_load_snapshot_rejects_corrupted_json() {
        let dir = temp_snapshot_dir();
        let store = FileStateStore::new(&dir)
            .await
            .expect("create file state store");
        let job_id = Uuid::new_v4();
        let path = dir.join(format!("{job_id}.json"));

        tokio::fs::write(&path, "{not-valid-json")
            .await
            .expect("write invalid snapshot");

        let err = store
            .load_snapshot(job_id)
            .await
            .expect_err("corrupted snapshot should fail");
        assert!(matches!(err, StoreError::Serialization(_)));

        tokio::fs::remove_dir_all(&dir)
            .await
            .expect("cleanup temp dir");
    }
}
