from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from vanet_env import VANETCommEnv
import os

def train_sb3_vanet():
    # Setup log directories
    log_dir = "SB3/logs/"
    eval_log_dir = os.path.join(log_dir, "eval/")
    os.makedirs(eval_log_dir, exist_ok=True)

    # Create monitored environment
    def make_env():
        return Monitor(VANETCommEnv())

    vec_env = make_vec_env(make_env, n_envs=1)
    eval_env = VANETCommEnv()

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(eval_log_dir, "best_model"),
        log_path=eval_log_dir,
        eval_freq=5000,
        n_eval_episodes=20,  # naikkan jadi 10–20 biar lebih representatif
        deterministic=True,  
        render=False,
    )


    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256,256]}
    )

    model.learn(total_timesteps=20000, tb_log_name="SAC_VANET", callback=eval_callback)
    model.save("model/sac_sb3_vanet")
    print("Model trained and saved as 'sac_sb3_vanet.zip'")
