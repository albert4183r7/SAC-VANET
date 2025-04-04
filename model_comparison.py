from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from vanet_env import VANETCommEnv  

# Inisialisasi environment
env = VANETCommEnv()

# Load kedua model
model_last = SAC.load("model/sac_sb3_vanet.zip", env=env)
model_best = SAC.load("SB3/logs/eval/best_model/best_model.zip", env=env)

# Evaluasi keduanya
mean_reward_last, std_reward_last = evaluate_policy(
    model_last, env, n_eval_episodes=20, deterministic=True, render=False)

mean_reward_best, std_reward_best = evaluate_policy(
    model_best, env, n_eval_episodes=20, deterministic=True, render=False)

# Cetak hasil perbandingan
print("📊 Perbandingan Inference:")
print(f"🔹 sac_sb3_vanet.zip (terakhir):     Mean Reward = {mean_reward_last:.2f} ± {std_reward_last:.2f}")
print(f"🔸 best_model.zip (eval):    Mean Reward = {mean_reward_best:.2f} ± {std_reward_best:.2f}")

# Kesimpulan otomatis
if mean_reward_best > mean_reward_last:
    print("\n✅ Gunakan best_model.zip untuk inference! (lebih unggul)")
elif mean_reward_last > mean_reward_best:
    print("\n✅ Gunakan sac_sb3_vanet.zip untuk inference! (reward akhir lebih baik)")
else:
    print("\n🤝 Keduanya seimbang, bisa pilih berdasarkan stabilitas atau ukuran.")
