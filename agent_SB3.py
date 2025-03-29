from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from vanet_env import VANETCommEnv
import gym

# Inisialisasi environment
env = VANETCommEnv()
check_env(env)

# Bungkus environment dengan Monitor untuk logging
env = Monitor(env)

# Buat environment untuk evaluasi
eval_env = Monitor(VANETCommEnv())

# Callback evaluasi dan early stop
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=45.0, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback
)

callback = CallbackList([eval_callback])

# Inisialisasi model SAC
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    buffer_size=100000,
    learning_starts=1000,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto"
)

# Training
model.learn(total_timesteps=100_000, callback=callback)

# Simpan model
model.save("vanet_sac_model_SB3")
