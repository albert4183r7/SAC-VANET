from stable_baselines3 import SAC

# Misalnya kamu sudah punya model
model = SAC.load("model/sac_sb3_vanet")

# Print semua bagian penting
print("=== Full SAC Model ===")
print(model)

print("\n=== Policy Structure ===")
print(model.policy)

print("\n=== Feature Extractor ===")
print(model.policy.features_extractor)

print("\n=== Actor Network ===")
print(model.policy.actor)

print("\n=== Critic 1 ===")
print(model.policy.critic)

print("\n=== Critic 2 ===")
print(model.policy.critic_target)
